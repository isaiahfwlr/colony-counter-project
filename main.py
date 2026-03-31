from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import tempfile
import os


app = FastAPI()


ROBOFLOW_API_KEY = "EXAMPLE_KEY"
MODEL_ID = "EXAMPLE_ID"


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)


@app.get("/")
def root():
    return {"status": "Colony Counter API is running"}


@app.post("/count")
async def count_colonies(file: UploadFile = File(...)):
    tmp_path = None

    try:
        # Read uploaded file
        image_bytes = await file.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file uploaded"}
            )

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name


        result = CLIENT.infer(tmp_path, model_id=MODEL_ID)

        predictions = result.get("predictions", [])
        colony_count = len(predictions)

        return {
            "colonies_detected": colony_count,
            "predictions": predictions
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
     
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
