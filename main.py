import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DIY Course Creator",
    version="2.0",
    description="API for creating courses and managing resources"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Hugging Face model
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Text generation function
def generate_text(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

# Web scraper
def scrape_website(url: str, query: str) -> Dict[str, Any]:
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        title = soup.title.string if soup.title else "No title found"
        
        # TODO: Make this more robust for different website structures
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        content_text = main_content.get_text(strip=True) if main_content else "No main content found"
        
        # Do some fancy TF-IDF magic
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query, content_text])
        relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return {
            "url": url,
            "title": title,
            "content": content_text[:500],  # Chop it off at 500 chars
            "relevance_score": relevance_score
        }
    finally:
        driver.quit()

# YouTube stuff
def get_video_id(url: str) -> str:
    try:
        return YouTube(url).video_id
    except Exception as e:
        logger.error(f"Oops, couldn't get video ID: {str(e)}")
        raise

def get_transcript(video_id: str) -> List[dict]:
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        logger.error(f"Transcript fetch failed: {str(e)}")
        raise

def segment_transcript(transcript: List[dict], segment_duration: int = 60) -> List[dict]:
    segments = []
    current_segment = {"text": "", "start": 0, "end": 0, "duration": 0}
    
    for entry in transcript:
        if current_segment["duration"] + entry["duration"] > segment_duration:
            current_segment["end"] = entry["start"]
            segments.append(current_segment)
            current_segment = {"text": entry["text"], "start": entry["start"], "end": entry["start"] + entry["duration"], "duration": entry["duration"]}
        else:
            current_segment["text"] += " " + entry["text"]
            current_segment["duration"] += entry["duration"]
            current_segment["end"] = entry["start"] + entry["duration"]
    
    if current_segment["text"]:
        segments.append(current_segment)
    
    return segments

def find_relevant_segments(segments: List[dict], query: str, top_k: int = 5) -> List[dict]:
    texts = [segment["text"] for segment in segments]
    texts.append(query)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    relevant_segments = [{"segment": segments[i], "score": float(cosine_similarities[i])} for i in top_indices]
    
    return relevant_segments

def analyze_continuous_actions(relevant_segments, max_gap=1.5):
    if not relevant_segments:
        return []

    sorted_segments = sorted(relevant_segments, key=lambda x: x['segment']['start'])
    continuous_segments = [sorted_segments[0]]
    
    for segment in sorted_segments[1:]:
        last_segment = continuous_segments[-1]
        if segment['segment']['start'] - last_segment['segment']['end'] <= max_gap:
            continuous_segments.append(segment)
        else:
            break
    
    return continuous_segments

def format_time(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"

def is_youtube_url(url):
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    youtube_regex += r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return bool(re.match(youtube_regex, url))

# Pydantic models
class UserStart(BaseModel):
    topic: str
    level: str
    time_period: str

class UserChangeResource(BaseModel):
    link: str
    main_topic: str
    sub_topic: str
    topic: str
    resource_type: str

# LangChain prompt
prompt_template_start = PromptTemplate(
    input_variables=["topic", "level", "time_period"],
    template="""
    Make a {time_period} course plan for {topic} at {level} level.
    Return it as a JSON object with weeks as keys, days as sub-keys, and topics as values.
    Each day should have a main topic and some subtopics.
    Example:
    {
      "1": {
        "1": {"Main Topic": ["Subtopic 1", "Subtopic 2"]},
        "2": {"Another Main Topic": ["Subtopic A", "Subtopic B"]}
      }
    }
    Make it comprehensive and suitable for the level.
    """
)

llm_chain_start = LLMChain(llm=hf_pipeline, prompt=prompt_template_start)

@app.post("/start/")
async def start(user_input: UserStart):
    try:
        inputs = {
            "topic": user_input.topic,
            "level": user_input.level,
            "time_period": user_input.time_period,
        }
        output = llm_chain_start.run(inputs)

        try:
            output_json = json.loads(output)
        except json.JSONDecodeError:
            logger.error("Couldn't parse the schedule")
            return {"error": "Schedule parsing failed. Try again?"}

        # Process the schedule and get resources
        final_schedule = process_schedule(output_json, user_input.level)

        return {
            "overview": output,
            "overview_json": output_json,
            "final_data": final_schedule
        }

    except Exception as e:
        logger.error(f"Uh oh, something went wrong: {str(e)}")
        return {"error": "Unexpected error. Give it another shot!"}

def process_schedule(schedule, level):
    weeks = []
    counter = 1

    for week_num, week_data in schedule.items():
        weekly_data = {"weekNumber": int(week_num), "days": []}
        
        for day_num, day_data in week_data.items():
            day_schedule = {}
            for main_topic, subtopics in day_data.items():
                day_schedule["title"] = main_topic
                day_schedule["dayNumber"] = int(day_num)
                day_schedule["timeSlots"] = []

                for subtopic in subtopics:
                    # FIXME: This time calculation is a bit hacky
                    time_slot = {
                        "subtitle": subtopic,
                        "time": f"{9 + 2*(int(day_num)-1)}:00 - {11 + 2*(int(day_num)-1)}:00",
                        "contents": [{
                            "id": counter,
                            "todo": [f"Learn {subtopic}"],
                            "resources": generate_resources(f"{main_topic}: {subtopic}", level)
                        }]
                    }
                    day_schedule["timeSlots"].append(time_slot)
                    counter += 1

            weekly_data["days"].append(day_schedule)
        
        weeks.append(weekly_data)

    return {"weeks": weeks}

async def scrape_website(url: str, query: str) -> Dict[str, Any]:
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    
    try:
        await asyncio.to_thread(driver.get, url)
        await asyncio.to_thread(WebDriverWait(driver, 10).until, EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        page_source = await asyncio.to_thread(lambda: driver.page_source)
        soup = BeautifulSoup(page_source, 'html.parser')
        
        title = soup.title.string if soup.title else "No title found"
        
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        content_text = main_content.get_text(strip=True) if main_content else "No main content found"
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query, content_text])
        relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Extract outgoing links
        outgoing_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')]
        
        return {
            "url": url,
            "title": title,
            "content": content_text[:500],
            "relevance_score": relevance_score,
            "outgoing_links": outgoing_links
        }
    finally:
        await asyncio.to_thread(driver.quit)

# PageRank-like algorithm
def calculate_page_rank(urls, damping_factor=0.85, num_iterations=100):
    G = nx.DiGraph()
    
    for url in urls:
        G.add_node(url["url"])
        for outgoing_link in url["outgoing_links"]:
            if outgoing_link in [u["url"] for u in urls]:
                G.add_edge(url["url"], outgoing_link)
    
    page_ranks = nx.pagerank(G, alpha=damping_factor, max_iter=num_iterations)
    
    return page_ranks

# Modified generate_resources function
async def generate_resources(topic, level):
    resources = []
    counter = 1

    search_url = f"https://www.google.com/search?q={topic} {level} tutorial"
    response = await asyncio.to_thread(requests.get, search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='g')

    urls_to_scrape = [result.find('a')['href'] for result in search_results[:5] if result.find('a')['href'].startswith('http')]

    # Use ThreadPoolExecutor for parallel scraping
    with ThreadPoolExecutor(max_workers=5) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, scrape_website, url, topic) for url in urls_to_scrape]
        scraped_data = await asyncio.gather(*tasks)

    # Calculate PageRank
    page_ranks = calculate_page_rank(scraped_data)

    # Sort scraped_data based on PageRank and relevance score
    sorted_data = sorted(scraped_data, key=lambda x: (page_ranks[x["url"]], x["relevance_score"]), reverse=True)

    for data in sorted_data[:2]:  # Take top 2 after sorting
        resource = {
            "id": counter,
            "for": topic,
            "link": data["url"],
            "description": data["title"]
        }
        
        if is_youtube_url(data["url"]):
            try:
                video_id = get_video_id(data["url"])
                transcript = get_transcript(video_id)
                yt = YouTube(data["url"])
                total_time = yt.length / 3600
                segment_duration = 60 if total_time <= 0.5 else (150 if total_time < 1.5 else (240 if total_time <= 2 else 400))
                segments = segment_transcript(transcript, segment_duration)
                relevant_segments = find_relevant_segments(segments, topic)
                final_segments = analyze_continuous_actions(relevant_segments)
                resource["description"] += f" (Watch from {format_time(final_segments[0]['segment']['start'])} to {format_time(final_segments[-1]['segment']['end'])})"
            except Exception as e:
                logger.error(f"YouTube processing hiccup: {str(e)}")
        
        resources.append(resource)
        counter += 1

    return resources

# Modified process_schedule function
async def process_schedule(schedule, level):
    weeks = []
    counter = 1

    for week_num, week_data in schedule.items():
        weekly_data = {"weekNumber": int(week_num), "days": []}
        
        for day_num, day_data in week_data.items():
            day_schedule = {}
            for main_topic, subtopics in day_data.items():
                day_schedule["title"] = main_topic
                day_schedule["dayNumber"] = int(day_num)
                day_schedule["timeSlots"] = []

                # Use asyncio.gather to process subtopics concurrently
                async_tasks = [generate_resources(f"{main_topic}: {subtopic}", level) for subtopic in subtopics]
                resources_list = await asyncio.gather(*async_tasks)

                for subtopic, resources in zip(subtopics, resources_list):
                    time_slot = {
                        "subtitle": subtopic,
                        "time": f"{9 + 2*(int(day_num)-1)}:00 - {11 + 2*(int(day_num)-1)}:00",
                        "contents": [{
                            "id": counter,
                            "todo": [f"Learn {subtopic}"],
                            "resources": resources
                        }]
                    }
                    day_schedule["timeSlots"].append(time_slot)
                    counter += 1

            weekly_data["days"].append(day_schedule)
        
        weeks.append(weekly_data)

    return {"weeks": weeks}

def generate_resources(topic, level):
    resources = []
    counter = 1

    # Do a quick Google search
    search_url = f"https://www.google.com/search?q={topic} {level} tutorial"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='g')

    for result in search_results[:2]:  # Just grab the top 2
        link = result.find('a')['href']
        if not link.startswith('http'):
            continue
        
        scraped_data = scrape_website(link, topic)
        
        resource = {
            "id": counter,
            "for": topic,
            "link": scraped_data["url"],
            "description": scraped_data["title"]
        }
        
        if is_youtube_url(link):
            try:
                video_id = get_video_id(link)
                transcript = get_transcript(video_id)
                yt = YouTube(link)
                total_time = yt.length / 3600  # Convert to hours
                segment_duration = 60 if total_time <= 0.5 else (150 if total_time < 1.5 else (240 if total_time <= 2 else 400))
                segments = segment_transcript(transcript, segment_duration)
                relevant_segments = find_relevant_segments(segments, topic)
                final_segments = analyze_continuous_actions(relevant_segments)
                resource["description"] += f" (Watch from {format_time(final_segments[0]['segment']['start'])} to {format_time(final_segments[-1]['segment']['end'])})"
            except Exception as e:
                logger.error(f"YouTube processing hiccup: {str(e)}")
        
        resources.append(resource)
        counter += 1

    return resources

@app.post("/change_resource/")
async def change_resource(user_input: UserChangeResource):
    try:
        if user_input.resource_type == "Video":
            search_query = f"{user_input.sub_topic}: {user_input.main_topic} ({user_input.topic}) tutorial"
            new_resources = generate_resources(search_query, "intermediate")  # Hardcoded to intermediate for now
            return {"change_resource": new_resources, "response": "Resource updated!"}
        else:
            return {"error": "Can't handle that resource type yet"}
    except Exception as e:
        logger.error(f"Resource change went sideways: {str(e)}")
        return {"error": "Something unexpected happened. Try again?"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
