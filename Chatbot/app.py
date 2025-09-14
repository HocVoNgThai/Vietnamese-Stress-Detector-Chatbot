import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import re
import torch

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except AttributeError:
    st.error("Lỗi: Vui lòng cung cấp GEMINI_API_KEY trong file .env")
    st.stop()

LABEL_MAP = {
    "LABEL_0": "Bình thường",
    "LABEL_1": "Lo lắng",
    "LABEL_2": "Bực bội",
    "LABEL_3": "Thất vọng",
    "LABEL_4": "Buồn/Cô đơn",
    "LABEL_5": "Mệt mỏi/Nghi ngờ bản thân",
    "LABEL_6": "Tuyệt vọng"
}
SEVERITY_LEVELS = {
    "Bình thường": 0,
    "Lo lắng": 1.5,
    "Bực bội": 1.5,
    "Thất vọng": 2.5,
    "Buồn/Cô đơn": 2.5,
    "Mệt mỏi/Nghi ngờ bản thân": 3.5,
    "Tuyệt vọng": 4.0
}
KEYWORD_BOOSTS = {
    "vô nghĩa": 1.0,
    "không muốn sống": 3.0,
    "muốn chết": 3.0,
    "hết hy vọng": 2.0,
    "mệt mỏi quá": 1.5,
    "áp lực": 1.0,
    "cô đơn": 1.5
}
try:
    tokenizer = AutoTokenizer.from_pretrained("../PhoBERT_emotion_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("../PhoBERT_emotion_classifier")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=128,
        truncation=True
    )
except Exception as e:
    st.error(f"Lỗi khi tải mô hình PhoBERT: {str(e)}. Vui lòng kiểm tra thư mục phobert_emotion_classifier.")
    classifier = None

def clean_text(text):

    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    text = word_tokenize(text, format="text")
    return text

def analyze_emotion(text):
    if not classifier:
        return "Không thể phân tích"
    cleaned = clean_text(text)
    with torch.no_grad():
        result = classifier(cleaned)
    return LABEL_MAP.get(result[0]['label'], "Không xác định")


RED_FLAG_KEYWORDS = ["tự tử", "tự sát", "chết", "kết thúc", "chấm dứt", "không muốn sống", "tạm biệt", "không thiết sống", "về trời"]

#Có thể thêm các thông tin liên hệ mà bạn muốn vào đây. Ví dụ thông tin của một bác sĩ tâm lý.
EMERGENCY_MESSAGE = """
Cảm ơn bạn đã chia sẻ. Mình nhận thấy bạn đang trải qua những cảm xúc rất nặng nề. 
Điều quan trọng nhất lúc này là bạn nhận được sự giúp đỡ ngay lập tức từ những người có chuyên môn. 

**Đừng ngần ngại, hãy liên hệ ngay:**
* **Đường dây nóng Ngày mai:** 096.306.1414 (8h-22h)
* **Tổng đài Quốc gia Bảo vệ Trẻ em:** 111 (24/7)
* **Các dịch vụ cấp cứu y tế gần nhất.**

Mình vẫn ở đây nếu bạn muốn tiếp tục chia sẻ, nhưng hãy ưu tiên liên hệ hỗ trợ chuyên môn nhé.
"""
ADVICE_MESSAGES = {
    "low": "Bạn dường như đang gặp một số khó khăn nhỏ. Hãy thử thư giãn bằng cách đi dạo, nghe nhạc hoặc chia sẻ với bạn bè. Nếu cần, bạn có thể trò chuyện với chuyên gia qua 1900555618.",
    "medium": "Bạn có vẻ đang trải qua áp lực tâm lý đáng kể. Hãy cân nhắc gặp bác sĩ tâm lý hoặc gọi 1900555618 để được tư vấn chuyên sâu. Bạn không cô đơn đâu!",
    "high": "Có vẻ bạn đang gặp khó khăn nghiêm trọng về tâm lý. Mình khuyên bạn nên liên hệ ngay với chuyên gia qua đường dây nóng 1900555618 hoặc gặp bác sĩ để được hỗ trợ kịp thời."
}
QIDS_QUESTIONS = [
    "Bạn có khó ngủ hoặc thức giấc thường xuyên không? (0: Không, 3: Rất nhiều)",
    "Bạn ngủ bao lâu mỗi đêm? (0: Vừa đủ, 3: Quá ít/quá nhiều)",
    "Bạn có thức giấc giữa đêm không? (0: Không, 3: Thường xuyên)",
    "Bạn có ngủ quá nhiều không? (0: Không, 3: Rất nhiều)",
    "Bạn cảm thấy buồn bã thế nào? (0: Không, 3: Rất buồn)",
    "Khẩu vị của bạn thay đổi ra sao? (0: Bình thường, 3: Thay đổi lớn)",
    "Cân nặng của bạn có thay đổi không? (0: Không, 3: Tăng/giảm nhiều)",
    "Bạn có khó tập trung không? (0: Không, 3: Rất khó)",
    "Bạn nghĩ về bản thân thế nào? (0: Tích cực, 3: Tiêu cực)",
    "Bạn có suy nghĩ về cái chết hoặc tự hại không? (0: Không, 3: Thường xuyên)",
    "Bạn có hứng thú với các hoạt động không? (0: Có, 3: Không)",
    "Mức năng lượng của bạn thế nào? (0: Cao, 3: Thấp)",
    "Bạn có cảm thấy chậm chạp không? (0: Không, 3: Rất chậm)",
    "Bạn có cảm thấy bồn chồn không? (0: Không, 3: Rất bồn chồn)",
    "Bạn có khó đưa ra quyết định không? (0: Không, 3: Rất khó)",
    "Bạn có cảm thấy tội lỗi không? (0: Không, 3: Rất nhiều)"
]
def check_for_red_flags(text):
    """Kiểm tra xem văn bản có chứa từ khóa nguy hiểm không."""
    return any(keyword in text.lower() for keyword in RED_FLAG_KEYWORDS)

def assess_overall_severity(history, red_flag_triggered):
    """Đánh giá mức độ tổng thể linh hoạt, ưu tiên từ khóa và red flag."""
    user_messages = [msg for msg in history if msg.role == "user"]
    num_user_msgs = len(user_messages)
    if num_user_msgs < 5 and not red_flag_triggered:  
        return None, None
    
    severity_scores = []
    for msg in user_messages:
        text = msg.parts[0].text.lower()
        emotion = analyze_emotion(text)
        score = SEVERITY_LEVELS.get(emotion, 0)
        keyword_boost = sum(KEYWORD_BOOSTS.get(kw, 0) for kw in KEYWORD_BOOSTS if kw in text)
        score += keyword_boost
        severity_scores.append(score)
    
    # Nếu có red flag, tăng mức nghiêm trọng ngay lập tức
    if red_flag_triggered:
        severity_scores.append(4.0)  # Tương đương "Tuyệt vọng"
    
    # Tính trung bình có trọng số
    weights = [1 + i * 0.2 for i in range(len(severity_scores))]
    weighted_avg = sum(score * weight for score, weight in zip(severity_scores, weights)) / sum(weights)
    
    # Kiểm tra streak nghiêm trọng
    recent_scores = severity_scores[-min(5, len(severity_scores)):]
    has_severe_streak = sum(1 for s in recent_scores if s >= 3.0) >= 3 or max(recent_scores) >= 4.0
    
    if weighted_avg >= 3.0 or has_severe_streak or red_flag_triggered:
        return "high", ADVICE_MESSAGES["high"]
    elif weighted_avg >= 1.5:
        return "medium", ADVICE_MESSAGES["medium"]
    elif weighted_avg > 0:
        return "low", ADVICE_MESSAGES["low"]
    return None, None

def compute_qids_score(scores):
    """Tính điểm QIDS-SR (0-27)."""
    total = sum(scores)
    if total <= 5:
        return "Không hoặc rất nhẹ. Hãy tiếp tục theo dõi sức khỏe tâm lý của bạn."
    elif total <= 10:
        return "Nhẹ. Hãy thử các hoạt động thư giãn như đi dạo, nghe nhạc, hoặc chia sẻ với bạn bè."
    elif total <= 15:
        return "Trung bình. Hãy cân nhắc gặp chuyên gia tâm lý qua 1900555618 để được tư vấn thêm."
    elif total <= 20:
        return "Nặng. Mình khuyên bạn nên gặp bác sĩ hoặc chuyên gia ngay để được hỗ trợ."
    else:
        return "Rất nặng. Vui lòng liên hệ chuyên gia hoặc đường dây nóng 1900555618 ngay lập tức."


SYSTEM_PROMPT = """
Bạn là "Trang Trang", trợ lý ảo đồng cảm, hỗ trợ sơ bộ về tâm lý. Nhiệm vụ:
1. Tạo không gian an toàn, đồng cảm, không phán xét.
2. Trả lời dựa trên bối cảnh cảm xúc, không lặp tên cảm xúc máy móc.
3. Linh hoạt đánh giá tổng thể, chỉ đưa gợi ý khi đủ dữ liệu hoặc dấu hiệu nghiêm trọng.
4. KHÔNG chẩn đoán bệnh. Chỉ gợi ý gặp chuyên gia nếu mức cao.
5. Nếu có red flag, nhấn mạnh hỗ trợ chuyên môn nhưng cho phép tiếp tục trò chuyện.
6. Trong chế độ khảo sát QIDS, chỉ hỏi câu hỏi và parse điểm (0-3), không trò chuyện khác.
7. Chỉ trả lời liên quan đến tin nhắn, KHÔNG tiết lộ bối cảnh nội bộ.
8. Tích hợp gợi ý khéo léo khi có đánh giá, ví dụ: "Có vẻ bạn đang gặp khó, hãy thử gặp chuyên gia nhé."

Ngôn ngữ nhẹ nhàng, phù hợp văn hóa Việt Nam.
"""
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest", #có thể thay mô hình khác tùy theo ý bạn
    system_instruction=SYSTEM_PROMPT
)

# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---
st.set_page_config(layout="centered", page_title="Chatbot Trò chuyện", page_icon = "🌕")
with st.sidebar.expander("ℹ️ Thông tin chung", expanded=False):
    st.markdown("### Thông tin chung")
    st.write("""
    Chatbot "Trang Trang" là một dự án cá nhân được phát triển nhằm hỗ trợ sơ bộ về tâm lý, sử dụng các công nghệ AI bao gồm 
    mô hình PhoBERT và API Google Generative AI. Mục tiêu của chatbot là cung cấp không gian an toàn để người dùng chia sẻ 
    cảm xúc và nhận gợi ý cơ bản. Đây là một ứng dụng thử nghiệm và không thay thế dịch vụ tư vấn chuyên môn từ các chuyên gia 
    tâm lý hoặc bác sĩ.
    """)
    
    st.markdown("---")
    st.write("**Phiên bản hiện tại:** 1.0")
    st.write("**Nhà phát triển:** Võ Nguyễn Thái Học")
    st.write("**Liên hệ:** vonguyenthaihocilt260@gmail.com")
    st.write("**Nguồn mở:** Dự án được lưu trữ tại GitHub repository.")
    
    st.markdown("---")
    st.markdown("### Thông Báo Pháp Lý")
    st.write("""
    Chatbot "Trang Trang" được phát triển như một dự án cá nhân và chỉ mang tính chất thông tin, thử nghiệm. Các phản hồi từ chatbot 
    có thể không chính xác, không đầy đủ hoặc chứa thiên kiến. Người dùng chịu trách nhiệm hoàn toàn cho bất kỳ quyết định hoặc 
    hành động nào dựa trên các phản hồi này và nên thực hiện với sự giám sát của con người để đảm bảo an toàn và phù hợp.

    Người tạo không chịu trách nhiệm về bất kỳ thiệt hại, mất mát hoặc hậu quả nào phát sinh từ việc sử dụng chatbot. Người dùng không nên nhập dữ liệu cá nhân, nhạy cảm hoặc được điều chỉnh (ví dụ: thông tin sức khỏe, tài chính) vào hệ thống. Bằng cách sử 
    dụng chatbot, bạn đồng ý rằng các đầu vào và phản hồi có thể được ghi lại để cải thiện hệ thống, 
    tuân thủ các quy định pháp luật hiện hành.

    Dự án này không liên kết hoặc được xác nhận bởi bất kỳ tổ chức thương mại hoặc nhà cung cấp API bên thứ ba nào.
    """)

    st.markdown("---")
    st.markdown("### Bản Quyền")
    st.write("""
    © 2025 HocVoNgThai. All rights reserved.
    Nội dung, mã nguồn và tài liệu liên quan đến chatbot "Trang Trang" được sở dụng cho mục đích học tập và nghiên cứu với điều kiện trích dẫn nguồn gốc. 
    Mong các bạn không tự ý sao chép, phân phối hoặc sử dụng lại cho mục đích thương mại.
    """)
    
    st.markdown("---")
    st.markdown("### Liên Kết Hữu Ích")
    st.link_button("LinkdIn", "https://linkedin.com/in/th1126/")
    st.link_button("Facebook", "https://facebook.com/th1126/")
    st.link_button("GitHub dự án", "https://github.com/HocVoNgThai/Vietnamese-Stress-Detector-Chatbot.git")


st.title("Trò chuyện cùng Trang Trang 🌕")
st.write("Mình tên là Trang Trang. Hãy chia sẻ bất cứ điều gì bạn đang nghĩ với mình nhé. Mình luôn sẵn lòng lắng nghe câu chuyện của bạn và cùng bạn vượt qua những khó khăn nè ❤️.")
st.markdown("---")

# Khởi tạo trạng thái
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
if "survey_mode" not in st.session_state:
    st.session_state.survey_mode = False
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "qids_scores" not in st.session_state:
    st.session_state.qids_scores = []
if "survey_prompted" not in st.session_state:
    st.session_state.survey_prompted = False
if "red_flag_triggered" not in st.session_state:
    st.session_state.red_flag_triggered = False

# Hiển thị các tin nhắn cũ
for message in st.session_state.chat_session.history:
    if message.role != "system":
        with st.chat_message(name="user" if message.role == "user" else "assistant"):
            st.markdown(message.parts[0].text)

# Ô nhập liệu và xử lý logic
if user_prompt := st.chat_input("Bạn đang cảm thấy thế nào?"):
    st.chat_message("user").markdown(user_prompt)

    # Kiểm tra từ khóa nguy hiểm
    if check_for_red_flags(user_prompt):
        st.session_state.red_flag_triggered = True
        st.chat_message("assistant").error(EMERGENCY_MESSAGE)
    
    # Phân tích cảm xúc từng lượt
    emotion_result = analyze_emotion(user_prompt)
    st.sidebar.info(f"Phân tích cảm xúc lượt này: **{emotion_result}**")

    # Đánh giá tổng thể linh hoạt
    overall_severity, overall_advice = assess_overall_severity(
        st.session_state.chat_session.history,
        st.session_state.red_flag_triggered
    )
    if overall_severity:
        st.sidebar.warning(f"Đánh giá sơ bộ tổng thể: Mức {overall_severity.upper()}")

    # Xử lý khảo sát QIDS nếu đang ở mode khảo sát
    if st.session_state.survey_mode:
        # Parse điểm từ phản hồi (dùng regex để tìm số 0-3)
        match = re.search(r'\b[0-3]\b', user_prompt)
        score = int(match.group()) if match else 0
        st.session_state.qids_scores.append(score)
        
        # Hỏi câu tiếp theo hoặc kết thúc khảo sát
        st.session_state.current_question += 1
        if st.session_state.current_question < len(QIDS_QUESTIONS):
            next_question = QIDS_QUESTIONS[st.session_state.current_question]
            st.chat_message("assistant").markdown(next_question)
        else:
            qids_result = compute_qids_score(st.session_state.qids_scores)
            st.chat_message("assistant").markdown(f"Dựa trên khảo sát, đánh giá sơ bộ: {qids_result}")
            st.session_state.survey_mode = False
            st.session_state.qids_scores = []
            st.session_state.current_question = 0
            st.session_state.survey_prompted = False
        st.stop() 

    # Gợi ý khảo sát nếu đủ lượt hoặc có red flag
    user_messages_count = len([m for m in st.session_state.chat_session.history if m.role == "user"])
    if (user_messages_count >= 10 or st.session_state.red_flag_triggered) and not st.session_state.survey_prompted and not st.session_state.survey_mode:
        st.session_state.survey_prompted = True
        survey_prompt = "Dựa trên cuộc trò chuyện, mình thấy có thể bạn đang gặp một số khó khăn. Bạn có muốn làm một khảo sát ngắn (16 câu) để đánh giá sơ bộ sức khỏe tâm lý không? (Có/Không)"
        st.chat_message("assistant").markdown(survey_prompt)
        st.stop()

    # Kiểm tra phản hồi khảo sát
    if st.session_state.survey_prompted and not st.session_state.survey_mode:
        if "có" in user_prompt.lower() or "yes" in user_prompt.lower():
            st.session_state.survey_mode = True
            first_question = QIDS_QUESTIONS[0]
            st.chat_message("assistant").markdown("Cảm ơn bạn! Bắt đầu khảo sát nhé. Trả lời theo mức độ 0-3. " + first_question)
            st.stop()
        else:
            st.session_state.survey_prompted = False

    # Tạo prompt cho Gemini
    enhanced_prompt = f"""
    [Bối cảnh phân tích nội bộ: Cảm xúc lượt này là '{emotion_result}'. 
    Đánh giá tổng thể: Mức '{overall_severity or "Chưa đánh giá"}'. 
    Red flag: {st.session_state.red_flag_triggered}. 
    Nếu có đánh giá tổng thể hoặc red flag, tích hợp gợi ý sau vào phản hồi khéo léo: '{overall_advice or "Chưa có gợi ý"}'. 
    KHÔNG lặp lại bối cảnh nội bộ.]

    Tin nhắn của người dùng: "{user_prompt}"
    """
    
    # Gửi tin nhắn đến Gemini
    with st.spinner("Trang Trang đang lắng nghe..."):
        response = st.session_state.chat_session.send_message(enhanced_prompt)
        st.chat_message("assistant").markdown(response.text)