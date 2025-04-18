from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from typing import Dict, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_video_transcript(video_id: str) -> Optional[Dict]:
    """
    YouTube 영상의 한글 자막을 가져오는 함수
    한글 자막이 없는 경우 자동 생성된 한글 자막을 가져옴

    Args:
        video_id (str): YouTube 영상 ID

    Returns:
        str: 자막 텍스트
    """
    try:
        # 사용 가능한 자막 목록 가져오기
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # 한글 자막 찾기
        try:
            # 수동 한글 자막 찾기
            transcript = transcript_list.find_transcript(['ko'])
            is_auto_generated = False
        except:
            try:
                # 자동 생성된 한글 자막 찾기
                transcript = transcript_list.find_transcript(['ko'], translation_languages=['ko'])
                is_auto_generated = True
            except:
                # 다른 언어의 자막을 한글로 번역
                transcript = transcript_list.find_transcript(['en']).translate('ko')
                is_auto_generated = True
        
        # 자막 데이터 가져오기
        transcript_data = transcript.fetch()
        
        # 자막을 텍스트로 변환
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript_data)
        
        return transcript_text
        
    except Exception as e:
        print(f"자막을 가져오는 중 오류가 발생했습니다: {str(e)}")
        return None

def format_time(seconds: float) -> str:
    """
    초를 시:분:초 형식으로 변환

    Args:
        seconds (float): 초

    Returns:
        str: 시:분:초 형식의 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def get_transcript_with_timestamps(video_id: str) -> Optional[str]:
    """
    시간 정보가 포함된 자막 텍스트를 반환하는 함수

    Args:
        video_id (str): YouTube 영상 ID

    Returns:
        Optional[str]: 시간 정보가 포함된 자막 텍스트
    """
    transcript_result = get_video_transcript(video_id)
    if not transcript_result:
        return None
    
    formatted_text = []
    for segment in transcript_result['segments']:
        start_time = format_time(segment['start'])
        text = segment['text']
        formatted_text.append(f"[{start_time}] {text}")
    
    return "\n".join(formatted_text)

def summarize_transcript(transcript_text: str, max_length: int = 500) -> Dict[str, str]:
    """
    OpenAI API를 사용하여 자막 내용을 요약하는 함수

    Args:
        transcript_text (str): 자막 텍스트
        max_length (int): 요약문의 최대 길이 (기본값: 500자)

    Returns:
        Dict[str, str]: {
            'summary': str,  # 전체 요약
            'key_points': str,  # 주요 포인트
            'topics': str  # 주요 주제
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # 요약 프롬프트 템플릿
    summary_template = """
    다음은 유튜브 영상의 자막 내용입니다. 이 내용을 바탕으로 다음 세 가지를 작성해주세요:

    1. 전체 내용 요약 (최대 {max_length}자)
    2. 주요 포인트 (5-7개)
    3. 주요 주제 (3-5개)

    자막 내용:
    {transcript}

    다음 형식으로 응답해주세요:
    {{
        "summary": "전체 내용 요약",
        "key_points": [
            "주요 포인트 1",
            "주요 포인트 2",
            ...
        ],
        "topics": [
            "주요 주제 1",
            "주요 주제 2",
            ...
        ]
    }}
    """

    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | llm

    try:
        result = chain.invoke({
            "transcript": transcript_text,
            "max_length": max_length
        })
        
        # 결과를 딕셔너리로 변환
        summary_dict = eval(result.content)
        
        # 리스트를 문자열로 변환
        summary_dict['key_points'] = "\n".join(f"• {point}" for point in summary_dict['key_points'])
        summary_dict['topics'] = "\n".join(f"• {topic}" for topic in summary_dict['topics'])
        
        return summary_dict
        
    except Exception as e:
        print(f"요약 생성 중 오류가 발생했습니다: {str(e)}")
        return {
            'summary': "요약을 생성할 수 없습니다.",
            'key_points': "주요 포인트를 추출할 수 없습니다.",
            'topics': "주요 주제를 추출할 수 없습니다."
        }

def analyze_video_content(video_id: str) -> Optional[Dict]:
    """
    영상의 자막을 가져와서 내용을 분석하는 함수

    Args:
        video_id (str): YouTube 영상 ID

    Returns:
        Dict: 요약 결과
    """
    # 자막 가져오기
    transcript_result = get_video_transcript(video_id)
    if not transcript_result:
        return None
    
    # 자막 내용 요약
    summary_result = summarize_transcript(transcript_result)
    
    return summary_result


if __name__ == "__main__":
    video_id = "9aWB1SLjowE"  # 예시로 유튜브 영상 ID를 사용
    result = analyze_video_content(video_id)
    if result:
        print(result)

