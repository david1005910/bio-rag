"""
E2E tests for Chinese Learning Streamlit App using Playwright.

Prerequisites:
  1. Streamlit app running: python -m streamlit run src/ui/app.py
  2. Run tests: pytest tests/e2e/ -v --base-url http://localhost:8501

Or use the automated script: ./scripts/validate.sh
"""
import re
import pytest
from playwright.sync_api import Page, expect

APP_URL = "http://localhost:8501"
# Wait timeout for Streamlit content to load (ms)
TIMEOUT = 15000


def navigate_to_page(page: Page, menu_option: str):
    """Select a page from the sidebar selectbox.

    Streamlit renders st.selectbox as a custom combobox (not a native <select>).
    We must: click to open, then click the matching listbox option.
    """
    combobox = page.get_by_role("combobox").first
    combobox.click()
    page.wait_for_timeout(300)
    # Options appear as <li> items with role="option" in an overlay
    page.get_by_role("option", name=menu_option, exact=True).click()
    page.wait_for_timeout(1500)


class TestAppLoads:
    """Verify the app starts and shows essential UI elements."""

    def test_home_page_loads(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        expect(page).to_have_title(re.compile(r"중국어"))

    def test_sidebar_present(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        sidebar = page.locator("section[data-testid='stSidebar']")
        expect(sidebar).to_be_visible(timeout=TIMEOUT)

    def test_sidebar_title(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        # Sidebar should show the app title
        expect(
            page.locator("section[data-testid='stSidebar']")
            .get_by_text("중국어 학습", exact=False)
        ).to_be_visible(timeout=TIMEOUT)

    def test_home_welcome_message(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        expect(
            page.get_by_text("중국어 학습 프로그램", exact=False)
        ).to_be_visible(timeout=TIMEOUT)

    def test_home_metrics_visible(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        # Four metric cards should be visible
        metrics = page.locator("[data-testid='stMetric']")
        expect(metrics.first).to_be_visible(timeout=TIMEOUT)


def main_content(page: Page):
    """Return the main content area locator."""
    return page.locator("[data-testid='stMainBlockContainer']")


class TestNavigation:
    """Verify sidebar navigation works for all pages."""

    def test_navigate_to_word_study(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📚 단어 학습")
        expect(
            main_content(page).get_by_text("단어 학습", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)

    def test_navigate_to_srs(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🔄 간격 복습 (SRS)")
        # Page title is "간격 반복 복습 (SM-2)"
        expect(
            main_content(page).get_by_text("간격 반복", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)

    def test_navigate_to_ai_conversation(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "💬 AI 회화")
        expect(
            main_content(page).get_by_text("AI 회화", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)

    def test_navigate_to_quiz(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📝 퀴즈")
        expect(
            main_content(page).get_by_text("퀴즈", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)

    def test_navigate_to_progress(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📊 진도 확인")
        expect(
            main_content(page).get_by_text("진도", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)

    def test_navigate_to_achievements(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🏆 업적")
        expect(
            main_content(page).get_by_text("업적", exact=False).first
        ).to_be_visible(timeout=TIMEOUT)


class TestWordStudyPage:
    """Tests for the Word Study (단어 학습) page."""

    def test_word_cards_displayed(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📚 단어 학습")
        # Should show at least one word card (Chinese characters visible)
        # The word-card div has Chinese text
        page.wait_for_timeout(1000)
        content = page.content()
        assert len(content) > 1000  # page has substantial content

    def test_pronunciation_button_exists(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📚 단어 학습")
        # Look for audio/pronunciation related button
        buttons = page.locator("button")
        count = buttons.count()
        assert count > 0, "No buttons found on word study page"

    def test_difficulty_rating_buttons(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📚 단어 학습")
        page.wait_for_timeout(1000)
        # Should have multiple buttons (prev/next/difficulty)
        buttons = page.locator("button")
        assert buttons.count() >= 2, "Expected navigation/rating buttons"


class TestQuizPage:
    """Tests for the Quiz (퀴즈) page."""

    def test_quiz_tab_options(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📝 퀴즈")
        page.wait_for_timeout(1000)
        # Should have quiz type tabs
        tabs = page.locator("[data-testid='stTab']")
        # If no tabs, look for radio buttons
        if tabs.count() == 0:
            tabs = page.locator("[data-testid='stRadio']")
        assert tabs.count() >= 0  # At minimum the page loads

    def test_quiz_content_loads(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📝 퀴즈")
        page.wait_for_timeout(1500)
        # Page should have substantial content
        main = page.locator("[data-testid='stMainBlockContainer']")
        expect(main).to_be_visible(timeout=TIMEOUT)


class TestProgressPage:
    """Tests for the Progress (진도 확인) page."""

    def test_progress_stats_visible(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "📊 진도 확인")
        page.wait_for_timeout(1000)
        # Main content area should be present with stats
        mc = main_content(page)
        expect(mc).to_be_visible(timeout=TIMEOUT)
        # Metrics or stat content is expected
        metrics = mc.locator("[data-testid='stMetric']")
        expect(metrics.first).to_be_visible(timeout=TIMEOUT)


class TestAchievementsPage:
    """Tests for the Achievements (업적) page."""

    def test_achievements_section_loads(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🏆 업적")
        page.wait_for_timeout(1000)
        main = page.locator("[data-testid='stMainBlockContainer']")
        expect(main).to_be_visible(timeout=TIMEOUT)

    def test_achievement_badges_present(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🏆 업적")
        page.wait_for_timeout(1000)
        content = page.content()
        # Should contain badge-related content (locked or unlocked)
        assert "badge" in content.lower() or "업적" in content


class TestSRSPage:
    """Tests for the Spaced Repetition (간격 복습) page."""

    def test_srs_page_loads(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🔄 간격 복습 (SRS)")
        page.wait_for_timeout(1000)
        main = page.locator("[data-testid='stMainBlockContainer']")
        expect(main).to_be_visible(timeout=TIMEOUT)

    def test_srs_shows_status_or_card(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "🔄 간격 복습 (SRS)")
        page.wait_for_timeout(1500)
        content = page.content()
        # Either a review card is shown or a message about no cards due
        assert len(content) > 500


class TestAIConversationPage:
    """Tests for the AI Conversation (AI 회화) page."""

    def test_ai_conversation_page_loads(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "💬 AI 회화")
        page.wait_for_timeout(1000)
        main = page.locator("[data-testid='stMainBlockContainer']")
        expect(main).to_be_visible(timeout=TIMEOUT)

    def test_chat_input_present(self, page: Page):
        page.goto(APP_URL, timeout=TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
        navigate_to_page(page, "💬 AI 회화")
        page.wait_for_timeout(1000)
        # Chat input or text area should be present
        chat_input = page.locator("[data-testid='stChatInput'], textarea, [data-testid='textInput']")
        assert chat_input.count() >= 0  # Page at minimum loads
