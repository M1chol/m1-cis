## Context Image Search
This utility provides you with advanced context to image search. 

## Requiered variablas
1. `GEMINI_API_KEY` - google gemini API key
2. `GOOGLE_API_KEY` - google custom search api key
3. `GOOGLE_CX` programmable search engine id

to aquire visit:
- [google gemini key](https://aistudio.google.com/apikey)
- [google custom search api](https://console.cloud.google.com/apis/api/customsearch.googleapis.com) -> Credentials
- [google custom search engine id](https://programmablesearchengine.google.com)

## Demo
You can check out the provided demo on [github](https://github.com/M1chol/m1-cis) in `./demo`. To get started:
1. `clone` [source repo](https://github.com/M1chol/m1-cis)
1. `cd` into `demo` dir
2. run `python -m venv .venv`
3. run `source .venv/bin/activate` or `venv\Scripts\activate` if on windows
3. run `pip install -r requierments.txt`
4. populate `.env.example` and rename to `.env`
5. run `streamlit run demo.py`

## Example
context: `Rheinmetall to Acquire German Naval Shipbuilder NVL`

| Result 1 | Result 2 |
| -------- | -------- | 
|![image](https://upload.wikimedia.org/wikipedia/commons/e/e1/USS_Dale_%28DLG-19%29_ready_for_launching_at_the_New_York_Shipbuilding_Company_on_27_July_1962_%28NH_98150%29.jpg?20100708150818)| ![image](https://images.pexels.com/photos/31389737/pexels-photo-31389737/free-photo-of-modern-naval-warship-docked-in-rotterdam-harbor.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)|