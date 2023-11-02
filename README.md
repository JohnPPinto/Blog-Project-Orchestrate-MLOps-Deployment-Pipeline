# Blog Project Orchestrate MLOps Deployment Pipeline
My article on MLOps deployment pipeline.

You can read the article for part 1 of the series by clicking [here](https://johnppinto.github.io/blog/posts/2023-11-01_deployment_part1/).<br>
Part 2 is currently in WIP.

This article uses the model [BLIP](https://github.com/salesforce/BLIP) developed by Salesforce, the model is used out-of-the-box. You can check the [demo notebook](https://github.com/JohnPPinto/Blog-Project-Orchestrate-MLOps-Deployment-Pipeline/blob/main/model-demo.ipynb) to understand how to use the model for making predictions using PyTorch.

The frontend and backend of the application are in the directory of the same name, it is built using Streamlit and FastAPI.

Both directories contain a module file called main.py, run the files simultaneously for the web application to work as needed.

```
streamlit run frontend/main.py
python backend/main.py
```
## Frontend

**Image to Text (Caption)**

![caption](https://github.com/JohnPPinto/Blog-Project-Orchestrate-MLOps-Deployment-Pipeline/assets/66053973/d320b46a-787e-406d-9af8-9334fcb79eb8)

**Image to Text (Visual Question Answer)**

![vqa](https://github.com/JohnPPinto/Blog-Project-Orchestrate-MLOps-Deployment-Pipeline/assets/66053973/d03f6361-5510-4e88-a829-fde48e3578ee)

## Backend

FastAPI Docs

![api](https://github.com/JohnPPinto/Blog-Project-Orchestrate-MLOps-Deployment-Pipeline/assets/66053973/33d8b92b-9efc-4367-80a5-fd01febfaa5b)
