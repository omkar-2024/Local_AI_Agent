<!-- install ollama_model from browser and then run from terminal-->
# install ollama_model from browser and then run from terminal
ollama pull qwen2.5:0.5b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b

<!-- start backend -->
# start backend
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

<!-- start frontend -->
# cd frontend
cd frontend
npm install
npm run dev

# download models folder and add in backend folder