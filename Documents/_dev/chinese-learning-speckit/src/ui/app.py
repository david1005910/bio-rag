"""
Streamlit 웹 인터페이스 - AI 기반 중국어 학습 프로그램
spec-kit 사양에 따른 풀스택 구현
"""

import streamlit as st
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.data_parser import ChineseDataParser
from src.core.lesson_manager import LessonManager
from src.core.progress_tracker import ProgressTracker
from src.ai.ai_tutor import ChineseAITutor
from src.ai.agents import OrchestratorAgent, EvalAgent
from src.speech.speech_handler import SpeechHandler
from src.learning.gamification import GamificationSystem, calculate_level, xp_progress_in_level
from src.learning.spaced_repetition import SpacedRepetitionSystem

st.set_page_config(
    page_title="중국어 학습",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS 스타일 ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.word-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    text-align: center;
    margin-bottom: 12px;
}
.word-card .chinese { font-size: 3rem; font-weight: bold; }
.word-card .pinyin  { font-size: 1.2rem; opacity: 0.9; margin-top: 4px; }
.xp-bar { height: 12px; border-radius: 6px; background: #e0e0e0; }
.xp-fill { height: 12px; border-radius: 6px; background: linear-gradient(90deg, #f093fb, #f5576c); }
.badge-unlocked   { background: #ffd700; border-radius: 12px; padding: 6px 12px; margin: 4px; display: inline-block; }
.badge-locked     { background: #e0e0e0; border-radius: 12px; padding: 6px 12px; margin: 4px; display: inline-block; opacity: 0.5; }
.metric-card      { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.correction-box   { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px; margin: 8px 0; }
.suggestion-box   { background: #d4edda; border-left: 4px solid #28a745; padding: 12px; border-radius: 4px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─── 초기화 ────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_resources():
    parser = ChineseDataParser()
    tracker = ProgressTracker('database/learning_progress.db')
    gamification = GamificationSystem(tracker)
    srs = SpacedRepetitionSystem(tracker)
    speech = SpeechHandler('audio_cache')
    ai_tutor = ChineseAITutor()
    orchestrator = OrchestratorAgent(tracker)
    eval_agent = EvalAgent()
    vocabulary = parser.load_hsk_words(level=1)
    lesson_manager = LessonManager(vocabulary)
    return {
        "parser": parser,
        "tracker": tracker,
        "gamification": gamification,
        "srs": srs,
        "speech": speech,
        "ai_tutor": ai_tutor,
        "orchestrator": orchestrator,
        "eval_agent": eval_agent,
        "vocabulary": vocabulary,
        "lesson_manager": lesson_manager,
    }


res = init_resources()


def get(key):
    return res[key]


# ─── 사이드바 ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🇨🇳 중국어 학습")
        st.markdown("---")

        # 레벨 & XP 표시
        level_info = get("gamification").get_level_info()
        level = level_info["level"]
        xp_cur = level_info["current_in_level"]
        xp_tot = level_info["xp_for_next_level"]
        pct = level_info["progress_percent"]
        streak = level_info["current_streak"]

        st.markdown(f"### Lv.{level} 학습자")
        st.markdown(f"""
<div class="xp-bar">
  <div class="xp-fill" style="width:{pct}%"></div>
</div>
<small>{xp_cur} / {xp_tot} XP</small>
""", unsafe_allow_html=True)

        if streak > 0:
            st.markdown(f"🔥 **{streak}일 연속 학습 중!**")

        st.markdown("---")

        menu = st.selectbox(
            "메뉴",
            ["🏠 홈", "📚 단어 학습", "🔄 간격 복습 (SRS)", "💬 AI 회화", "📝 퀴즈", "📊 진도 확인", "🏆 업적"],
            label_visibility="collapsed",
        )
        return menu


# ─── 홈 ───────────────────────────────────────────────────────────────────────
def show_home():
    st.title("🇨🇳 중국어 학습 프로그램에 오신 것을 환영합니다!")
    st.markdown("AI 기반 · 간격 반복 · 게임화 학습")

    # 오늘 학습 처리
    streak_result = get("gamification").update_streak()
    if not streak_result.get("already_done"):
        st.success(f"🔥 오늘도 학습 시작! 현재 연속: {streak_result.get('current_streak', 1)}일")

    stats = get("tracker").get_statistics()
    level_info = get("gamification").get_level_info()

    # 주요 지표
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("레벨", f"Lv.{level_info['level']}", delta=f"{level_info['total_xp']} XP")
    with c2:
        st.metric("마스터 단어", f"{stats['mastered_words']}개", delta=f"총 {stats['total_words_learned']}개")
    with c3:
        st.metric("연속 학습", f"{stats['current_streak']}일", delta=f"최고 {stats['longest_streak']}일")
    with c4:
        st.metric("평균 퀴즈", f"{stats['average_quiz_score']:.1f}점")

    # 학습 시작 버튼
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📚 단어 학습 시작", use_container_width=True, type="primary"):
            st.session_state.menu = "📚 단어 학습"
            st.rerun()
    with c2:
        due = len(get("srs").get_due_cards(50))
        if st.button(f"🔄 복습하기 ({due}개 대기)", use_container_width=True):
            st.session_state.menu = "🔄 간격 복습 (SRS)"
            st.rerun()
    with c3:
        if st.button("💬 AI와 대화하기", use_container_width=True):
            st.session_state.menu = "💬 AI 회화"
            st.rerun()

    # 학습 곡선
    curve = get("tracker").get_learning_curve(30)
    if curve:
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame(curve, columns=["날짜", "세션수", "평균점수"])
        df["평균점수"] = df["평균점수"].fillna(0)
        fig = px.area(df, x="날짜", y="평균점수", title="최근 30일 학습 추이", markers=True,
                      color_discrete_sequence=["#764ba2"])
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ─── 단어 학습 ─────────────────────────────────────────────────────────────────
def show_vocabulary_lesson():
    st.header("📚 단어 학습")

    vocab = get("vocabulary")
    total_lessons = max(1, len(vocab) // 10)

    c1, c2 = st.columns([2, 1])
    with c1:
        lesson_num = st.slider("레슨 선택", 0, total_lessons - 1, 0,
                               help=f"총 {total_lessons}개 레슨 (레슨당 10단어)")
    with c2:
        words_per_lesson = st.select_slider("단어 수", [5, 10, 15, 20], value=10)

    if st.button("📖 레슨 시작", type="primary"):
        words = get("lesson_manager").get_lesson(lesson_num, words_per_lesson)
        if not words:
            st.warning("이 레슨에 단어가 없습니다.")
            return
        session_id = get("tracker").start_session(lesson_num)
        st.session_state.lesson_words = words
        st.session_state.lesson_session_id = session_id
        st.session_state.lesson_learned = {}

    if "lesson_words" not in st.session_state:
        st.info("레슨 번호를 선택하고 시작 버튼을 누르세요.")

        # 오늘의 단어 미리보기 (3개)
        st.subheader("오늘의 미리보기 단어")
        for w in vocab[:3]:
            st.markdown(f"""
<div class="word-card">
  <div class="chinese">{w['simplified']}</div>
  <div class="pinyin">{w.get('pinyin', '')}</div>
</div>""", unsafe_allow_html=True)
        return

    words = st.session_state.lesson_words
    st.success(f"레슨 진행 중 — {len(words)}개 단어")

    for i, word in enumerate(words):
        with st.expander(f"{i+1}. **{word['simplified']}** — {word.get('pinyin', '')}", expanded=(i == 0)):
            col_info, col_actions = st.columns([3, 1])

            with col_info:
                st.markdown(f"""
<div class="word-card">
  <div class="chinese">{word['simplified']}</div>
  <div class="pinyin">{word.get('pinyin', '')}</div>
</div>""", unsafe_allow_html=True)
                defs = word.get("definitions", [])
                if defs:
                    st.markdown(f"**의미:** {' / '.join(defs)}")
                if word.get("traditional") and word["traditional"] != word["simplified"]:
                    st.caption(f"번체자: {word['traditional']}")

            with col_actions:
                # TTS 발음
                if st.button("🔊 발음", key=f"tts_{i}"):
                    audio = get("speech").text_to_speech(word["simplified"])
                    if audio and os.path.exists(audio):
                        st.audio(audio)
                    else:
                        st.info(f"발음: {word.get('pinyin', '')}")

                # 학습 완료 체크
                checked = st.checkbox("✅ 외웠어요", key=f"learned_{i}",
                                       value=st.session_state.lesson_learned.get(i, False))
                if checked and not st.session_state.lesson_learned.get(i):
                    st.session_state.lesson_learned[i] = True
                    get("tracker").update_word_mastery(word, True)
                    xp_result = get("gamification").award_xp("word_learned")
                    st.success(f"+{xp_result['xp_gained']} XP")
                    if xp_result.get("leveled_up"):
                        st.balloons()
                        st.success(f"🎉 레벨업! Lv.{xp_result['level']}")

    # 레슨 완료
    st.markdown("---")
    learned_count = sum(1 for v in st.session_state.lesson_learned.values() if v)
    st.progress(learned_count / len(words), text=f"{learned_count}/{len(words)}개 완료")

    if st.button("🎓 레슨 완료", type="primary"):
        get("tracker").end_session(
            st.session_state.lesson_session_id,
            learned_count, None
        )
        # 업적 체크
        new_achievements = get("gamification").check_achievements()
        if new_achievements:
            for ach in new_achievements:
                st.balloons()
                st.success(f"🏆 업적 달성: {ach['icon']} {ach['name']} — {ach['description']}")

        st.success(f"레슨 완료! {learned_count}개 단어를 학습했습니다.")
        del st.session_state.lesson_words
        del st.session_state.lesson_session_id
        st.session_state.lesson_learned = {}
        st.rerun()


# ─── SRS 복습 ─────────────────────────────────────────────────────────────────
def show_srs_review():
    st.header("🔄 간격 반복 복습 (SM-2)")
    st.caption("과학적 간격 반복으로 최적 시점에 복습합니다.")

    due_cards = get("srs").get_due_cards(20)

    if not due_cards:
        st.success("🎉 오늘 복습할 단어가 없습니다! 내일 다시 확인해보세요.")
        srs_stats = get("srs").get_statistics()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("전체 단어", srs_stats["total_words"])
        with c2:
            st.metric("마스터 완료", srs_stats["mastered"])
        return

    st.info(f"📋 오늘 복습할 단어: **{len(due_cards)}개**")

    if "srs_idx" not in st.session_state:
        st.session_state.srs_idx = 0
        st.session_state.srs_correct = 0
        st.session_state.srs_total = 0
        st.session_state.srs_show_answer = False

    idx = st.session_state.srs_idx
    if idx >= len(due_cards):
        st.success(f"🎉 복습 완료! {st.session_state.srs_correct}/{st.session_state.srs_total} 정답")
        get("gamification").award_xp("daily_goal_met")
        if st.button("복습 다시 시작"):
            for k in ["srs_idx", "srs_correct", "srs_total", "srs_show_answer"]:
                del st.session_state[k]
            st.rerun()
        return

    card = due_cards[idx]
    total = len(due_cards)

    st.progress((idx) / total, text=f"{idx+1}/{total}")

    # 단어 카드
    st.markdown(f"""
<div class="word-card" style="max-width:400px; margin:auto;">
  <div class="chinese">{card['simplified']}</div>
  <div class="pinyin">{card.get('pinyin', '')}</div>
</div>""", unsafe_allow_html=True)

    if not st.session_state.srs_show_answer:
        if st.button("💡 정답 보기", use_container_width=True):
            st.session_state.srs_show_answer = True
            st.rerun()
    else:
        defs = card.get("definitions", [])
        st.markdown(f"### 의미: {' / '.join(defs)}")
        st.markdown("**얼마나 기억했나요?**")

        cols = st.columns(4)
        quality_labels = [
            ("😫 완전 까먹음", 0, "red"),
            ("😕 어렵게 기억", 2, "orange"),
            ("🙂 약간 망설임", 4, "blue"),
            ("😄 완벽!", 5, "green"),
        ]
        for col, (label, quality, _) in zip(cols, quality_labels):
            with col:
                if st.button(label, use_container_width=True):
                    result = get("srs").process_review(card["simplified"], quality)
                    if quality >= 3:
                        st.session_state.srs_correct += 1
                        get("gamification").award_xp("quiz_correct")
                    st.session_state.srs_total += 1
                    st.session_state.srs_idx += 1
                    st.session_state.srs_show_answer = False
                    st.rerun()


# ─── AI 회화 ──────────────────────────────────────────────────────────────────
def show_conversation():
    st.header("💬 AI 회화 연습")
    st.caption("AI 튜터와 중국어로 대화하며 실력을 키우세요!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_corrections = []
        st.session_state.chat_session_id = get("tracker").start_session(0, "conversation")
        st.session_state.chat_turn = 0

    # 대화 기록 표시
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and i < len(st.session_state.chat_corrections):
                correction = st.session_state.chat_corrections[i // 2]
                if correction.get("corrections"):
                    with st.expander("📝 교정 사항"):
                        for c in correction["corrections"]:
                            st.markdown(f"""
<div class="correction-box">
  ❌ <b>원문:</b> {c.get('original', '')} → ✅ <b>교정:</b> {c.get('corrected', '')}<br>
  💡 {c.get('explanation', '')}
</div>""", unsafe_allow_html=True)
                if correction.get("suggestions"):
                    for s in correction["suggestions"]:
                        st.markdown(f'<div class="suggestion-box">💡 {s}</div>', unsafe_allow_html=True)

    # 사용자 입력
    user_input = st.chat_input("중국어 또는 한국어로 입력하세요...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("AI 튜터가 응답 중..."):
            response = get("ai_tutor").chat_practice(user_input)

        ai_text = response.get("response", "很好！继续加油！")
        pinyin = response.get("response_pinyin", "")
        translation = response.get("response_translation", "")

        display = ai_text
        if pinyin:
            display += f"\n*{pinyin}*"
        if translation:
            display += f"\n**{translation}**"

        st.session_state.chat_history.append({"role": "assistant", "content": display})
        st.session_state.chat_corrections.append(response)

        # 저장
        st.session_state.chat_turn += 1
        get("tracker").save_conversation_turn(
            st.session_state.chat_session_id,
            st.session_state.chat_turn,
            user_input, ai_text,
            str(response.get("corrections", [])),
            str(response.get("suggestions", [])),
        )
        get("gamification").award_xp("conversation_turn")
        st.rerun()

    # 대화 초기화
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 대화 초기화"):
            for k in ["chat_history", "chat_corrections", "chat_session_id", "chat_turn"]:
                if k in st.session_state:
                    del st.session_state[k]
            get("ai_tutor").conversation_history = []
            st.rerun()
    with col2:
        if st.button("📖 문법 설명 요청"):
            if st.session_state.chat_history:
                last_user = next(
                    (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "user"),
                    None
                )
                if last_user:
                    with st.spinner("문법 설명 생성 중..."):
                        explanation = get("orchestrator").tutor_agent.explain_grammar(last_user)
                    st.info(explanation.get("structure", ""))
                    for gp in explanation.get("grammar_points", []):
                        st.markdown(f"- **{gp.get('point', '')}**: {gp.get('explanation', '')}")


# ─── 퀴즈 ─────────────────────────────────────────────────────────────────────
def show_quiz():
    st.header("📝 퀴즈")

    vocab = get("vocabulary")
    stats = get("tracker").get_statistics()

    # 퀴즈 설정
    if "quiz_data" not in st.session_state:
        c1, c2, c3 = st.columns(3)
        with c1:
            count = st.slider("문제 수", 5, 20, 10)
        with c2:
            lesson_start = st.slider("레슨 시작", 0, max(0, len(vocab) // 10 - 1), 0)
        with c3:
            st.write("")  # spacer

        if st.button("🎯 퀴즈 시작", type="primary"):
            word_slice = vocab[lesson_start * 10: lesson_start * 10 + 50]
            if not word_slice:
                word_slice = vocab
            recent_scores = []
            # 최근 점수 기반 adaptive
            recent_scores = [stats.get("average_quiz_score", 70)] * 3

            questions = get("eval_agent").generate_adaptive_quiz(word_slice, recent_scores, count)
            session_id = get("tracker").start_session(lesson_start, "quiz")
            st.session_state.quiz_data = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_session_id = session_id
            st.rerun()
        return

    questions = st.session_state.quiz_data
    st.info(f"총 {len(questions)}문제")

    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['question']}")
            q_type = q.get("type", "translation")

            if q_type == "fill_blank":
                answer = st.text_input(f"답 입력 #{i+1}", key=f"q_{i}", label_visibility="collapsed")
            else:
                options = q.get("options", [])
                if options:
                    answer = st.radio(f"선택 #{i+1}", options, key=f"q_{i}", label_visibility="collapsed")
                else:
                    answer = st.text_input(f"답 입력 #{i+1}", key=f"q_{i}", label_visibility="collapsed")

            st.session_state.quiz_answers[i] = answer
            st.markdown("---")

        submitted = st.form_submit_button("✅ 채점하기", type="primary")

    if submitted:
        score = 0
        total_points = 0
        results = []

        for i, q in enumerate(questions):
            user_answer = st.session_state.quiz_answers.get(i, "")
            eval_result = get("eval_agent").evaluate_answer(q, user_answer)
            results.append(eval_result)
            score += eval_result["score"]
            total_points += q.get("points", 5)

        percentage = (score / total_points * 100) if total_points > 0 else 0

        if percentage >= 100:
            st.balloons()
        get("tracker").end_session(
            st.session_state.quiz_session_id, 0, percentage
        )
        xp_gained = get("gamification").award_xp("quiz_correct", extra_multiplier=percentage / 100)
        if percentage >= 100:
            get("gamification").award_xp("quiz_perfect")

        # 결과 표시
        st.markdown(f"## 결과: {score}/{total_points}점 ({percentage:.1f}%)")
        if percentage >= 80:
            st.success("🎉 훌륭해요!")
        elif percentage >= 60:
            st.warning("💪 조금 더 연습해봐요!")
        else:
            st.error("📚 단어를 더 복습해보세요!")

        st.info(f"+{xp_gained['xp_gained']} XP 획득!")

        for i, (q, r) in enumerate(zip(questions, results)):
            if r["is_correct"]:
                st.success(f"Q{i+1} ✓ {r['feedback']}")
            else:
                st.error(f"Q{i+1} ✗ {r['feedback']}")
            if r.get("explanation"):
                st.caption(r["explanation"])

        # 새 퀴즈
        if st.button("🔄 새 퀴즈"):
            for k in ["quiz_data", "quiz_answers", "quiz_session_id"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


# ─── 진도 확인 ─────────────────────────────────────────────────────────────────
def show_progress():
    st.header("📊 학습 진도")

    stats = get("tracker").get_statistics()
    level_info = get("gamification").get_level_info()

    # 주요 지표
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("레벨", f"Lv.{level_info['level']}")
    with c2:
        st.metric("총 XP", f"{level_info['total_xp']}")
    with c3:
        st.metric("학습 시간", f"{stats['total_study_hours']:.1f}시간")
    with c4:
        st.metric("마스터 단어", f"{stats['mastered_words']}개")

    # XP 진행바
    st.markdown(f"**다음 레벨까지**: {level_info['current_in_level']} / {level_info['xp_for_next_level']} XP")
    st.progress(level_info["progress_percent"] / 100)

    st.markdown("---")

    # 단어 마스터리 분포
    col_chart, col_review = st.columns([2, 1])

    with col_chart:
        curve = get("tracker").get_learning_curve(30)
        if curve:
            import pandas as pd
            import plotly.express as px
            df = pd.DataFrame(curve, columns=["날짜", "세션수", "평균점수"])
            df["평균점수"] = df["평균점수"].fillna(0)
            fig = px.bar(df, x="날짜", y="세션수", title="30일 학습 세션",
                         color_discrete_sequence=["#764ba2"])
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("아직 데이터가 없습니다. 학습을 시작해보세요!")

    with col_review:
        st.subheader("복습 대기 단어")
        review_words = get("tracker").get_words_for_review(8)
        if review_words:
            for w in review_words:
                st.markdown(f"- **{w[0]}** ({w[2]}) — *{w[1]}*")
        else:
            st.success("복습할 단어가 없습니다! 🎉")

    # 데이터 에이전트 인사이트
    st.markdown("---")
    st.subheader("💡 학습 인사이트")
    data_agent = get("orchestrator").data_agent
    if data_agent:
        analysis = data_agent.analyze_progress()
        for insight in analysis.get("insights", []):
            st.info(insight)
        st.subheader("📋 추천 사항")
        for rec in analysis.get("recommendations", []):
            st.markdown(f"- {rec}")


# ─── 업적 ─────────────────────────────────────────────────────────────────────
def show_achievements():
    st.header("🏆 업적")

    achievements = get("gamification").get_all_achievements()

    if not achievements:
        st.info("아직 업적이 없습니다. 학습을 시작하면 업적을 달성할 수 있어요!")
        return

    # 카테고리별 분류
    categories = {"words": "📖 단어", "streak": "🔥 연속", "score": "🎯 점수",
                  "time": "⏱️ 시간", "special": "⭐ 특별"}

    for cat_key, cat_name in categories.items():
        cat_achievements = [a for a in achievements if a.get("category") == cat_key]
        if not cat_achievements:
            continue

        st.subheader(cat_name)
        cols = st.columns(min(4, len(cat_achievements)))
        for i, ach in enumerate(cat_achievements):
            with cols[i % len(cols)]:
                css_class = "badge-unlocked" if ach["unlocked"] else "badge-locked"
                icon = ach.get("icon", "🏅")
                st.markdown(f"""
<div class="{css_class}" style="text-align:center; width:100%;">
  <div style="font-size:2rem;">{icon}</div>
  <b>{ach['name']}</b><br>
  <small>{ach['description']}</small>
  {'<br><small>✅ 달성!</small>' if ach['unlocked'] else ''}
</div>""", unsafe_allow_html=True)

    unlocked = sum(1 for a in achievements if a["unlocked"])
    st.markdown("---")
    st.metric("달성한 업적", f"{unlocked} / {len(achievements)}")
    st.progress(unlocked / len(achievements) if achievements else 0)


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    menu = render_sidebar()

    # 세션에서 메뉴 오버라이드 처리
    if "menu" in st.session_state:
        menu = st.session_state.pop("menu")

    if menu == "🏠 홈":
        show_home()
    elif menu == "📚 단어 학습":
        show_vocabulary_lesson()
    elif menu == "🔄 간격 복습 (SRS)":
        show_srs_review()
    elif menu == "💬 AI 회화":
        show_conversation()
    elif menu == "📝 퀴즈":
        show_quiz()
    elif menu == "📊 진도 확인":
        show_progress()
    elif menu == "🏆 업적":
        show_achievements()


if __name__ == "__main__":
    main()
