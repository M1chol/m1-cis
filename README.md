> [!WARNING]
> This package is in development

## Context Image Search
This utility provides you with advanced context to image search. 

## How it works
1. Decomposes given text (context) into search querries using `gemini-2.5-flash-lite`
2. Searches the querries on curated list of cc licensed images using google images
3. Analyzes the images for visual match using ML
4. Returns the best matching images

## Requiered variablas
1. `GEMINI_API_KEY` - google gemini API key
2. `GOOGLE_API_KEY` - google custom search api key
3. `GOOGLE_CX` programmable search engine id

to aquire visit:
- [google gemini key](https://aistudio.google.com/apikey)
- [google custom search api](https://console.cloud.google.com/apis/api/customsearch.googleapis.com) -> Credentials
- [google custom search engine id](https://programmablesearchengine.google.com)

## Example
context: `Rheinmetall to Acquire German Naval Shipbuilder NVL`

resulting images:
![image](https://upload.wikimedia.org/wikipedia/commons/e/e1/USS_Dale_%28DLG-19%29_ready_for_launching_at_the_New_York_Shipbuilding_Company_on_27_July_1962_%28NH_98150%29.jpg?20100708150818)
![image](https://images.pexels.com/photos/31389737/pexels-photo-31389737/free-photo-of-modern-naval-warship-docked-in-rotterdam-harbor.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)