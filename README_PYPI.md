## Context Image Search
This utility provides you with advanced context to image search. 

## Requiered variablas
1. `GOOGLE_API_KEY` - google custom search api key
2. `GOOGLE_CX` programmable search engine id

As `GOOGLE_CX` you can use `37d934a53d9a8443e` which i set up to work with those domains.
```
www.flickr.com/*
www.freepik.com/*
*.openverse.org/*
*.nappy.co/*
www.wikipedia.org/*
commons.wikimedia.org/*
*.unsplash.com/*
www.pexels.com/*
*.pixabay.com/*
```
Or create your own [google custom search engine](https://programmablesearchengine.google.com)

Please enable those API's for your key:
- [Generative Language API (Gemini API)](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com)
- [Custom Search API](https://console.cloud.google.com/marketplace/product/google/customsearch.googleapis.com)

## Demo
[See github page](https://github.com/M1chol/m1-cis?tab=readme-ov-file#demo)

## Example
context: `Rheinmetall to Acquire German Naval Shipbuilder NVL`

| Result 1 | Result 2 |
| -------- | -------- | 
|![image](https://upload.wikimedia.org/wikipedia/commons/e/e1/USS_Dale_%28DLG-19%29_ready_for_launching_at_the_New_York_Shipbuilding_Company_on_27_July_1962_%28NH_98150%29.jpg?20100708150818)| ![image](https://images.pexels.com/photos/31389737/pexels-photo-31389737/free-photo-of-modern-naval-warship-docked-in-rotterdam-harbor.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)|
