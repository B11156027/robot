from openai import OpenAI
from fish_audio_sdk import WebSocketSession, TTSRequest
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import io
import threading
import time
import serial
import cv2
import requests
import threading

# Firebase 相關導入
from datetime import datetime
from firebase_admin import db
import firebase_init

# =========== 人臉追蹤＋馬達控制 ==========
ESP32_CAM_STREAM = "http://192.168.0.2:81/stream"
SERVO_SERVER_IP = "192.168.0.20"
SERVO_MIN = 1
SERVO_MAX = 179

# 設定 OpenAI 客戶端
client = OpenAI(api_key="     ")

# 建立 TTS WebSocket 連接
sync_websocket = WebSocketSession("78790b183cab4a4e8d3076ce74056b5c")

# 建立語音識別器
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# 全域變數控制語音識別
listening = False
recognition_thread = None

# Firebase 設定
device_id = "voice_assistant_001"

arduino = serial.Serial('COM4', 9600, timeout=1)

# 全域變數 - 互譯模式控制
translate_mode = False

def upload_to_firebase(user_input, ai_response, conversation_type="voice"):
    """
    將對話內容上傳到 Firebase
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        conversation_data = {
            "user_input": user_input,
            "ai_response": ai_response,
            "conversation_type": conversation_type,  # "voice" 或 "translate"
            "timestamp": timestamp,
            "device_id": device_id,
            "response_length": len(ai_response)
        }
        
        conversation_ref = db.reference(f'ai_conversations/{device_id}')
        conversation_ref.push(conversation_data)
        
        print(f"✅ 對話已上傳到 Firebase: {timestamp}")
        
    except Exception as e:
        print(f"❌ Firebase 上傳錯誤: {str(e)}")

def upload_system_status(status, message=""):
    """
    上傳系統狀態到 Firebase
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        status_data = {
            "status": status,
            "message": message,
            "timestamp": timestamp,
            "device_id": device_id
        }
        
        status_ref = db.reference(f'system_status/{device_id}')
        status_ref.push(status_data)
        
    except Exception as e:
        print(f"❌ 系統狀態上傳錯誤: {str(e)}")

def face_tracking_thread(stop_flag):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(ESP32_CAM_STREAM)
    CENTER_DEG = 70
    SENS = 1.10
    ALPHA = 0.35
    SEND_EPS = 2
    SEND_INTERVAL = 0.07
    HEAD_MIN, HEAD_MAX = SERVO_MIN, SERVO_MAX 
    CROP_TOP, CROP_BOT = 0.20, 0.20
    CROP_LEFT, CROP_RIGHT = 0.00, 0.00
    
    if not cap.isOpened():
        print("串流無法開啟，請檢查串流網址或網路！")
        return
    
    CROP_BOT_RATIO = 0.20
    print("人臉追蹤啟動！即時追蹤中...")
    
    while not stop_flag["stop"]:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        bottom = int(h * 0.6)
        frame = frame[:bottom, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        angle = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_center_x = x + w // 2
            W = frame.shape[1]

            err = (face_center_x - W/2) / (W/2)
            GAMMA = 1.3
            GAIN  = 2.5
            err_nl = (abs(err) ** GAMMA) * (1 if err >= 0 else -1)

            half_span = (SERVO_MAX - SERVO_MIN) / 2
            angle = int(CENTER_DEG + GAIN * err_nl * half_span)
            angle = max(SERVO_MIN, min(SERVO_MAX, angle))
            break
            
        if angle is not None:
            url = f'http://{SERVO_SERVER_IP}/servo?angle={angle}'
            try:
                requests.get(url, timeout=0.1)
            except Exception as e:
                pass
        cv2.imshow("ESP32-CAM 人臉追蹤", frame)
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("人臉追蹤已結束")

def chat_with_gpt(prompt, conversation_history, model="gpt-3.5-turbo"):
    """
    與 ChatGPT 進行對話並返回回應文字
    """
    try:
        conversation_history.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            max_tokens=100,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
        
    except Exception as e:
        return f"發生錯誤: {str(e)}"

def translate_text(text, target_language):
    """
    翻譯文字功能
    """
    try:
        if target_language == "english":
            system_prompt = "You are a translator. Translate the following Chinese text to English. Only return the English translation, no explanations."
        else:
            system_prompt = "You are a translator. Translate the following English text to Traditional Chinese. Only return the Chinese translation, no explanations."
            
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=200,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"翻譯錯誤: {str(e)}"

def detect_language(text):
    """
    簡單的語言檢測
    """
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_chars = sum(1 for char in text if char.isalpha() and char.isascii())
    
    if chinese_chars > english_chars:
        return "chinese"
    else:
        return "english"

def text_to_speech(text, mouth=True):
    try:
        def stream():
            for line in text.split():
                yield line + " "

        tts_request = TTSRequest(
            text="",
            reference_id="075f601e716547829a2397c18ff72317",
            temperature=0.7,
            top_p=0.7,
        )

        audio_data = b""
        for chunk in sync_websocket.tts(tts_request, stream(), backend="speech-1.5"):
            audio_data += chunk
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))

        # 嘴巴同步
        playing_flag = [True]
        def mouth_move_thread():
            try:
                arduino.write(b'S')
                while playing_flag[0]:
                    arduino.write(b'S')
                    time.sleep(0.3)
            except Exception as e:
                print(f"[嘴巴同步錯誤] {e}")

        t = None
        if mouth:
            t = threading.Thread(target=mouth_move_thread)
            t.start()

        play(audio)

        if mouth:
            playing_flag[0] = False
            arduino.write(b'E')
            if t:
                t.join()

    except Exception as e:
        print(f"TTS 錯誤: {str(e)}")

def speech_to_text():
    """
    語音轉文字
    """
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2
    recognizer.non_speaking_duration = 2.0
    
    while True:
        try:
            print("🎤 請開始說話...（停頓1.5秒自動結束）")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=None)
            text = recognizer.recognize_google(audio, language="zh-TW")
            return text
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            continue

def continuous_voice_recognition(conversation_history):
    """
    持續語音識別功能
    """
    global listening, translate_mode
    
    print("\n🎤 語音識別已啟動！")
    print("💡 語音指令：")
    print("   - 說 '互譯模式' 開啟中英互譯")
    print("   - 說 '聊天模式' 關閉互譯，回到正常對話")
    print("   - 說 '結束對話' 或 '再見' 可結束程式")
    print("   - 說 '清除記憶' 可清除對話記錄")
    print("   - 說 '查看記錄' 可顯示對話歷史\n")
    
    upload_system_status("voice_mode", "語音模式啟動")
    
    while listening:
        try:
            user_input = speech_to_text()
            
            if user_input is None:
                continue
            
            # 檢查特殊指令
            if any(cmd in user_input for cmd in ['結束對話', '再見', '掰掰']):
                print("👋 再見！")
                farewell_msg = "再見啦！"
                text_to_speech(farewell_msg)
                upload_to_firebase(user_input, farewell_msg, "voice")
                upload_system_status("stopped", "用戶結束對話")
                return "quit"
                
            elif any(cmd in user_input for cmd in ['互譯模式', '翻譯模式', '開啟翻譯']):
                translate_mode = True
                status_msg = "已開啟互譯模式"
                print(f"🔄 {status_msg}")
                text_to_speech(status_msg)
                upload_to_firebase(user_input, status_msg, "voice")
                continue
                
            elif any(cmd in user_input for cmd in ['聊天模式', '關閉翻譯', '停止翻譯']):
                translate_mode = False
                status_msg = "已關閉互譯模式"
                print(f"🔄 {status_msg}")
                text_to_speech(status_msg)
                upload_to_firebase(user_input, status_msg, "voice")
                continue
                
            elif any(cmd in user_input for cmd in ['清除記憶', '清除對話', '重新開始']):
                system_messages = [msg for msg in conversation_history if msg["role"] == "system"]
                conversation_history.clear()
                conversation_history.extend(system_messages)
                print("✅ 對話記憶已清除！\n")
                clear_msg = "記憶已清除"
                text_to_speech(clear_msg)
                upload_to_firebase(user_input, clear_msg, "voice")
                continue
                
            elif any(cmd in user_input for cmd in ['查看記錄', '顯示記錄', '對話記錄']):
                print("\n=== 對話記錄 ===")
                for msg in conversation_history:
                    if msg["role"] == "system":
                        print(f"[系統] {msg['content'][:50]}...")
                    elif msg["role"] == "user":
                        print(f"[你] {msg['content']}")
                    elif msg["role"] == "assistant":
                        print(f"[AI] {msg['content']}")
                print("==================\n")
                history_msg = "記錄已顯示"
                text_to_speech(history_msg)
                upload_to_firebase(user_input, history_msg, "voice")
                continue
            
            # 正常對話處理
            print(f"🗣️ 你說: {user_input}")
            
            if translate_mode:
                # 互譯模式
                detected_lang = detect_language(user_input)
                print(f"🔍 檢測語言: {'中文' if detected_lang == 'chinese' else '英文'}")
                
                if detected_lang == "chinese":
                    # 中文 → 英文
                    translation = translate_text(user_input, "english")
                    print(f"📝 英文翻譯: {translation}")
                    response_text = translation
                    upload_to_firebase(user_input, f"[中→英] {translation}", "translate")
                else:
                    # 英文 → 中文
                    translation = translate_text(user_input, "chinese")
                    print(f"📝 中文翻譯: {translation}")
                    response_text = translation
                    upload_to_firebase(user_input, f"[英→中] {translation}", "translate")
                
                print("🔊 正在播放翻譯結果...")
                text_to_speech(response_text)
                print("✅ 翻譯播放完成！\n")
                
            else:
                # 正常聊天模式
                gpt_response = chat_with_gpt(user_input, conversation_history)
                print(f"🤖 AI回答: {gpt_response}\n")
                
                upload_to_firebase(user_input, gpt_response, "voice")
                
                print("🔊 正在播放語音...")
                text_to_speech(gpt_response)
                print("✅ 語音播放完成！\n")
            
        except KeyboardInterrupt:
            print("\n⚠️ 語音識別被中斷")
            upload_system_status("interrupted", "語音識別被中斷")
            break
        except Exception as e:
            print(f"❌ 語音識別過程發生錯誤: {e}")
            upload_system_status("error", f"語音識別錯誤: {str(e)}")
            continue
    
    return None

def get_user_mode():
    """
    讓用戶選擇模式
    """
    print("\n📝 請選擇模式：")
    print("1. 互譯模式（中英文互譯）")
    print("2. 語音聊天模式（AI對話）")
    
    while True:
        choice = input("請選擇 (1-2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("❌ 無效選擇，請輸入 1 或 2")

def chat_and_speak():
    """
    純語音助手主函數
    """
    global listening, translate_mode
    
    print("=== ChatGPT 純語音助手 ===")
    print("🎤 純語音操作，支援中英互譯和AI對話")
    print("📊 Firebase 整合：所有對話都會上傳到雲端")
    
    upload_system_status("started", "純語音助手啟動")
    
    # 選擇初始模式
    initial_mode = get_user_mode()
    
    if initial_mode == "1":
        translate_mode = True
        print("🔄 已設定為互譯模式")
        print("   🇨🇳 說中文 → 翻譯成英文")
        print("   🇺🇸 說英文 → 翻譯成中文\n")
    else:
        translate_mode = False
        print("💬 已設定為語音聊天模式")
        print("   🤖 與AI進行正常對話\n")
    
    # 系統提示詞
    conversation_history = []
    system_prompt = "你是一個友善且樂於助人的 AI 助手，回答問題時請使用繁體中文，語氣要親切自然。請將每次回答控制在30個中文字以內，簡潔明瞭。"
    conversation_history.append({"role": "system", "content": system_prompt})
    
    print("✅ 純語音操作模式")
    print("✅ 支援中英互譯")
    print("✅ 支援AI對話")
    print("✅ Firebase 雲端記錄")
    print("✅ 人臉追蹤功能\n")
    
    # 開始語音識別
    listening = True
    continuous_voice_recognition(conversation_history)

# 執行程式
if __name__ == "__main__":
    try:
        # 啟動人臉追蹤線程
        stop_flag = {"stop": False}
        tracking_thread = threading.Thread(target=face_tracking_thread, args=(stop_flag,))
        tracking_thread.daemon = True
        tracking_thread.start()

        # 檢查麥克風可用性
        print("🔧 正在檢查麥克風...")
        with sr.Microphone() as source:
            pass
        print("✅ 麥克風檢查通過！\n")
        
        # 啟動主程式
        chat_and_speak()
        
    except Exception as e:
        print(f"❌ 麥克風初始化失敗: {e}")
        upload_system_status("error", f"麥克風初始化失敗: {str(e)}")
        print("⚠️ 無法啟動純語音模式")
        
    finally:
        # 當主程式結束時終止人臉追蹤
        stop_flag["stop"] = True
        if 'tracking_thread' in locals():
            tracking_thread.join()
        upload_system_status("shutdown", "程式正常關閉")