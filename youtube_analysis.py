import streamlit as st
import requests
from streamlit_player import st_player
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
import os
from dotenv import load_dotenv
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import plotly.express as px
from context_analysis import analyze_video_content

# .env 파일에서 환경 변수 로드
load_dotenv()

# YouTube Data API 설정
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Streamlit 애플리케이션 제목
st.title("YouTube Video Analysis")

# URL 입력받기
video_url = st.text_input("YouTube 동영상 URL을 입력하세요:")

# 댓글 수 조절 슬라이더 추가
max_comments = st.slider(
    "수집할 댓글 수를 선택하세요",
    min_value=10,
    max_value=1000,
    value=500,
    step=50,
    help="더 많은 댓글을 수집하면 분석에 시간이 더 걸릴 수 있습니다."
)

def get_dislike_count(video_id):
    # API URL 설정
    api_url = f"https://returnyoutubedislikeapi.com/votes?videoId={video_id}"

    # API 호출
    response = requests.get(api_url)

    # 응답 처리
    if response.status_code == 200:
        data = response.json()
        if "dislikes" in data:
            return data["dislikes"]
        else:
            return "Dislike count not available"
    else:
        return f"Failed to fetch data. Status code: {response.status_code}"

# YouTube 동영상 ID 추출 함수
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path.startswith("/shorts/"):  # Shorts URL 처리
            return parsed_url.path.split("/")[2]  # '/shorts/{video_id}'에서 video_id 추출
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

# 동영상 정보 가져오기 함수
def get_video_info(video_id):
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()

        if "items" in response and len(response["items"]) > 0:
            video = response["items"][0]
            title = video["snippet"]["title"]
            description = video["snippet"]["description"]
            view_count = video["statistics"].get("viewCount", "N/A")
            like_count = video["statistics"].get("likeCount", "N/A")

            return {
                "title": title,
                "description": description,
                "view_count": view_count,
                "like_count": like_count
            }
        else:
            return None
    except HttpError as e:
        st.error(f"YouTube API 오류 발생: {e}")
        return None

# 댓글 가져오기 함수
def get_video_comments(video_id, max_comments=100):
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            remaining_comments = max_comments - len(comments)
            current_max_results = min(100, remaining_comments)

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=current_max_results,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments[:max_comments]
    except HttpError as e:
        st.error(f"YouTube API 오류 발생: {e}")
        return []
    except Exception as e:
        st.error(f"오류 발생: {e}")
        return []

# 감성 분석 함수
def analyze_sentiment(comments: List[Dict], video_id: str) -> Dict:
    if not OPENAI_API_KEY:
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return {}

    # 영상 내용 분석
    video_content = analyze_video_content(video_id)
    video_summary = video_content.get('summary', '') if video_content else ''
    video_key_points = video_content.get('key_points', '') if video_content else ''
    video_topics = video_content.get('topics', '') if video_content else ''

    comment_texts = [comment['text'] for comment in comments]
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # 프롬프트 템플릿 설정
    template = """
    다음은 유튜브 영상에 대한 댓글들입니다. 이 댓글들이 영상에 대해 전반적으로 긍정적인지 부정적인지 분석해주세요.
    특히 논란이 되거나 이슈가 될 수 있는 댓글들을 찾아주세요. 만약 논란이 되는 댓글이 없다면 controversial_comments 키에 빈 리스트를 반환해주세요.
    댓글 비율은 숫자로만 반환해주세요.

    영상 내용 요약:
    {video_summary}

    주요 포인트:
    {video_key_points}

    주요 주제:
    {video_topics}

    댓글들:
    {comments}

    다음 딕셔너리 형식으로 응답해주세요:
    {{
        "overall_sentiment": "긍정/부정/중립",
        "positive_ratio": "긍정적 댓글 비율 (%)",
        "negative_ratio": "부정적 댓글 비율 (%)",
        "neutral_ratio": "중립 댓글 비율 (%)",
        "positive_comments": [
            "주요 긍정적 댓글 1",
            "주요 긍정적 댓글 2",
            "주요 긍정적 댓글 3"
        ],
        "negative_comments": [
            "주요 부정적 댓글 1",
            "주요 부정적 댓글 2",
            "주요 부정적 댓글 3"
        ],
        "neutral_comments": [
            "주요 중립 댓글 1",
            "주요 중립 댓글 2",
            "주요 중립 댓글 3"
        ],
        "controversial_comments": [
            {{
                "comment": "이슈가 되는 댓글 내용",
                "reason": "이슈가 되는 이유",
                "impact_level": "높음/중간/낮음"
            }},
            {{
                "comment": "이슈가 되는 댓글 내용",
                "reason": "이슈가 되는 이유",
                "impact_level": "높음/중간/낮음"
            }}
        ],
        "analysis": "종합적인 분석 내용"
    }}
    반드시 파이썬 딕셔너리 형식으로 응답해주세요. 따옴표는 작은따옴표(')를 사용해주세요.
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    result = chain.invoke({
        "comments": "\n".join(comment_texts),
        "video_summary": video_summary,
        "video_key_points": video_key_points,
        "video_topics": video_topics
    })

    try:
        analysis_result = eval(result.content)
        return {
            'raw_analysis': analysis_result,
            'comment_count': len(comments)
        }
    except Exception as e:
        st.error(f"딕셔너리 변환 오류: {e}")
        return {
            'raw_analysis': result.content,
            'comment_count': len(comments),
            'error': '딕셔너리 변환 실패'
        }

# Streamlit UI
if video_url:
    st.write("### 동영상 미리보기:")

    video_id = extract_video_id(video_url)
    if video_id:
        # Shorts URL을 포함한 동영상 URL 생성
        if "shorts" in video_url:
            video_url = f"https://www.youtube.com/watch?v={video_id}"  # Shorts URL을 일반 URL로 변환
        # 동영상 미리보기 출력
        st_player(video_url)

        video_info = get_video_info(video_id)
        if video_info:
            st.write("### 동영상 정보:")
            st.write(f"**제목:** {video_info['title']}")
            st.write(f"**조회수:** {int(video_info['view_count']):,}")
            st.write(f"**좋아요 수:** {int(video_info['like_count']):,}")

            dislike_count = get_dislike_count(video_id)
            if isinstance(dislike_count, int):
                st.write(f"**싫어요 수:** {dislike_count:,}")
            else:
                st.write(f"**싫어요 수:** {dislike_count}")
            
            # 좋아요와 싫어요 비율 계산
            like_count = int(video_info['like_count'])
            total_votes = like_count + (dislike_count if isinstance(dislike_count, int) else 0)
            if total_votes > 0:
                like_percentage = (like_count / total_votes) * 100
                dislike_percentage = (dislike_count / total_votes) * 100 if isinstance(dislike_count, int) else 0
                
                # 좋아요/싫어요 비율을 하나의 통합된 슬라이더로 표현
                st.write("### 좋아요/싫어요 비율")
                st.markdown(f"""
                    <div style='text-align: center; margin-bottom: 10px;'>
                        <span style='color: #28a745; font-weight: bold;'>좋아요: {like_percentage:.2f}%</span>
                        <span style='margin: 0 10px;'>|</span>
                        <span style='color: #dc3545; font-weight: bold;'>싫어요: {dislike_percentage:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # 통합된 진행 바
                st.progress(like_percentage / 100, text=f"좋아요 {like_percentage:.2f}% | 싫어요 {dislike_percentage:.2f}%")
            else:
                st.write("좋아요와 싫어요 데이터가 충분하지 않습니다.")

            # 영상 내용 분석 결과 표시
            st.write("### 영상 내용 분석")
            with st.spinner('영상 내용을 분석하고 있습니다...'):
                video_content = analyze_video_content(video_id)
                
                if video_content:
                    # 영상 요약
                    with st.expander("📝 영상 요약", expanded=True):
                        st.markdown(video_content.get('summary', '요약을 불러올 수 없습니다.'))
                    
                    # 주요 포인트
                    with st.expander("🎯 주요 포인트", expanded=True):
                        key_points = video_content.get('key_points', '')
                        if key_points:
                            # 줄넘김 기호를 기준으로 분리
                            points = key_points.split('\n')
                            for point in points:
                                if point.strip():  # 빈 줄이 아닌 경우에만 표시
                                    st.markdown(f"{point.strip()}")
                        else:
                            st.write("주요 포인트를 불러올 수 없습니다.")
                    
                    # 주요 주제
                    with st.expander("🏷️ 주요 주제", expanded=True):
                        topics = video_content.get('topics', '')
                        if topics:
                            # 줄넘김 기호를 기준으로 분리
                            topic_list = topics.split('\n')
                            st.markdown("**주요 주제:**")
                            for topic in topic_list:
                                if topic.strip():  # 빈 줄이 아닌 경우에만 표시
                                    st.markdown(f"{topic.strip()}")
                        else:
                            st.write("주요 주제를 불러올 수 없습니다.")
                else:
                    st.error("영상 내용을 분석할 수 없습니다.")

            st.write("### 댓글 수집 및 감성 분석:")
            with st.spinner('댓글을 수집하고 있습니다...'):
                comments = get_video_comments(video_id, max_comments=max_comments)
            
            if comments:
                st.write(f"수집된 댓글 수: {len(comments)}")
                with st.spinner('댓글 감성 분석 중...'):
                    sentiment_analysis = analyze_sentiment(comments, video_id)

                if 'raw_analysis' in sentiment_analysis:
                    analysis = sentiment_analysis['raw_analysis']

                    st.write("### 감성 분석 결과:")

                    # 전체 감성
                    st.markdown(f"**전체 감성:** {analysis.get('overall_sentiment', 'N/A')}")

                    # 감성 비율을 원형 차트로 표시
                    sentiment_data = {
                        '긍정': float(analysis.get('positive_ratio', '0').strip('%')),
                        '부정': float(analysis.get('negative_ratio', '0').strip('%')),
                        '중립': float(analysis.get('neutral_ratio', '0').strip('%'))
                    }
                    
                    # 원형 차트로 감성 비율 표시
                    st.write("#### 댓글 감성 분포")
                    fig = px.pie(
                        values=list(sentiment_data.values()),
                        names=list(sentiment_data.keys()),
                        title='댓글 감성 분포',
                        color=list(sentiment_data.keys()),
                        color_discrete_map={'긍정': '#007bff', '부정': '#dc3545', '중립': '#6c757d'}
                    )
                    st.plotly_chart(fig)

                    # 공통 의견 분석 결과 표시
                    st.write("#### 주요 의견 분석")
                    common_opinions = analysis.get('common_opinions', [])
                    other_opinions = analysis.get('other_opinions', {})
                    
                    # 주요 의견들을 카드 형태로 표시
                    for opinion in common_opinions:
                        with st.expander(f"📌 {opinion['opinion']} ({opinion['ratio']})"):
                            st.write("**대표 댓글:**")
                            for comment in opinion['example_comments']:
                                st.write(f"- {comment}")
                    
                    # 기타 의견 표시
                    if other_opinions:
                        with st.expander(f"📌 기타 의견 ({other_opinions.get('ratio', '0%')})"):
                            st.write("**대표 댓글:**")
                            for comment in other_opinions.get('example_comments', []):
                                st.write(f"- {comment}")

                    # 긍정적/부정적/중립적 댓글을 탭으로 구분하여 표시
                    st.write("#### 상세 댓글 분석")
                    tab1, tab2, tab3 = st.tabs(["긍정적 댓글", "부정적 댓글", "중립적 댓글"])
                    
                    with tab1:
                        st.write("**긍정적 댓글:**")
                        positive_comments = analysis.get('positive_comments', [])
                        if positive_comments:
                            for idx, comment in enumerate(positive_comments, start=1):
                                st.write(f"{idx}. {comment}")
                        else:
                            st.write("긍정적 댓글이 없습니다.")

                    with tab2:
                        st.write("**부정적 댓글:**")
                        negative_comments = analysis.get('negative_comments', [])
                        if negative_comments:
                            for idx, comment in enumerate(negative_comments, start=1):
                                st.write(f"{idx}. {comment}")
                        else:
                            st.write("부정적 댓글이 없습니다.")
                    
                    with tab3:
                        st.write("**중립적 댓글:**")
                        neutral_comments = analysis.get('neutral_comments', [])
                        if neutral_comments:
                            for idx, comment in enumerate(neutral_comments, start=1):
                                st.write(f"{idx}. {comment}")
                        else:
                            st.write("중립적 댓글이 없습니다.")

                    # 논란이 되는 댓글
                    st.write("#### 논란이 되는 댓글")
                    controversial_comments = analysis.get('controversial_comments', [])
                    if controversial_comments:
                        for idx, comment_data in enumerate(controversial_comments, start=1):
                            with st.expander(f"⚠️ 논란 댓글 {idx}"):
                                comment = comment_data.get('comment', 'N/A')
                                reason = comment_data.get('reason', 'N/A')
                                impact_level = comment_data.get('impact_level', 'N/A')
                                st.markdown(f"**댓글 내용:** {comment}")
                                st.markdown(f"**논란 이유:** {reason}")
                                st.markdown(f"**영향 수준:** {impact_level}")
                    else:
                        st.write("논란이 되는 댓글이 없습니다.")

                    # 종합 분석 내용
                    st.write("#### 종합 분석")
                    st.markdown(f"{analysis.get('analysis', 'N/A')}")
                else:
                    st.error("감성 분석 결과를 처리하는 데 실패했습니다.")
            else:
                st.error("댓글을 가져오는 데 실패했습니다.")
        else:
            st.error("동영상 정보를 가져오는 데 실패했습니다.")
    else:
        st.error("유효한 YouTube 동영상 ID를 URL에서 추출할 수 없습니다.")
else:
    st.write("YouTube URL을 입력하면 동영상 정보를 분석합니다.")