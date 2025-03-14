import streamlit as st
from openai import OpenAI

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # .env 파일에 저장된 키

class TravelPlanner:
    def __init__(self):
        # DeepSeek 공식 API 연결
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,  # DeepSeek에서 발급받은 API 키
            base_url="https://api.deepseek.com"  # DeepSeek 공식 API 엔드포인트
        )
        self.model = "deepseek-chat"  # DeepSeek의 공식 모델 이름 (예: deepseek-chat)

    def process_request(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # 적절한 창의성 조절
                max_tokens=1024,  # 최대 토큰 설정
                stream=False,  # DeepSeek 공식 API는 스트리밍 방식은 미지원 (2024-2025 기준)
            )

            # 응답 텍스트 가져오기
            result_text = response.choices[0].message.content
            return result_text

        except Exception as e:
            return f"❌ Error: {str(e)}"

    @staticmethod
    def get_system_prompts():
        return {
            "Trip Itinerary": """You are a travel expert who creates detailed and personalized trip itineraries.
        Follow these guidelines:
        1. Start with an overview of the destination
        2. Include a day-by-day breakdown of activities
        3. Suggest must-visit attractions and hidden gems
        4. Provide recommendations for local cuisine and dining
        5. Include transportation tips and options
        6. Add cultural or historical context for key locations
        7. Offer packing tips based on the destination's climate""",
            "Travel Tips": """You are a seasoned traveler who provides practical advice for smooth trips.
        Provide tips on:
        1. Best times to visit specific destinations
        2. Budgeting and saving money while traveling
        3. Navigating local customs and etiquette
        4. Staying safe and healthy during travel
        5. Packing efficiently for different types of trips
        6. Finding affordable accommodations and flights
        7. Making the most of layovers and short trips""",
            "Destination Recommendations": """You are a travel guide who suggests destinations based on user preferences.
        Consider:
        1. The traveler's interests (e.g., adventure, relaxation, culture)
        2. Budget constraints
        3. Preferred climate and season
        4. Travel duration
        5. Group size and demographics (e.g., family, solo, couple)
        6. Accessibility and travel restrictions
        7. Unique experiences or events happening at the destination""",
        }

    @staticmethod
    def get_example_prompts():
        return {
            "Trip Itinerary": {
                "placeholder": """Examples:
1. "Plan a 5-day trip to Japan focusing on culture and food"
2. "Create a 7-day itinerary for a family vacation in Italy"
3. "Suggest a 3-day weekend getaway for adventure lovers in Costa Rica"
4. "Design a 10-day road trip across the American Southwest"
5. "Plan a romantic 4-day trip to Paris"

Your request:""",
                "default": "Plan a 7-day trip to Japan focusing on culture and food",
            },
            "Travel Tips": {
                "placeholder": """Ask for travel tips or advice.
Examples:
1. "What are the best ways to save money while traveling in Europe?"
2. "How can I stay safe while traveling solo in South America?"
3. "What should I pack for a two-week trip to Southeast Asia?"
4. "What are some tips for traveling with young children?"
5. "How do I handle language barriers in non-English-speaking countries?""",
                "default": "What are the best ways to save money while traveling in Europe?",
            },
            "Destination Recommendations": {
                "placeholder": """Describe your preferences for destination suggestions.
Examples:
1. "I want a relaxing beach vacation with good food and clear water"
2. "I'm looking for an adventurous trip with hiking and wildlife"
3. "Suggest a cultural destination with historical sites and museums"
4. "I need a budget-friendly destination for a family of four"
5. "Where can I go for a romantic getaway with stunning views?""",
                "default": "I want a relaxing beach vacation with good food and clear water",
            },
        }


def main():
    st.set_page_config(
        page_title="DeepSeek Travel Assistant",
        page_icon="✈️",
        layout="wide",
    )

    st.title("✈️ DeepSeek Travel Assistant")
    st.markdown("""
        Powered by DeepSeek API (chat model).
    """)

    system_prompts = TravelPlanner.get_system_prompts()
    example_prompts = TravelPlanner.get_example_prompts()

    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Trip Itinerary", "Travel Tips", "Destination Recommendations"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Mode**: {mode}")
    st.sidebar.markdown("**Mode Description:**")
    st.sidebar.markdown(system_prompts[mode].replace("\n", "\n\n"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Input for {mode}")
        user_prompt = st.text_area(
            "Enter your travel preferences or questions:",
            height=300,
            placeholder=example_prompts[mode]["placeholder"],
            value=example_prompts[mode]["default"],
        )

        process_button = st.button("✈️ Process", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Output")
        output_container = st.container()

    if process_button:
        if user_prompt.strip():
            with st.spinner("Planning your trip..."):
                assistant = TravelPlanner()
                result = assistant.process_request(system_prompts[mode], user_prompt)
                output_container.markdown(result)
        else:
            st.warning("⚠️ Please enter some input!")


if __name__ == "__main__":
    main()
