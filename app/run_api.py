# Simple launcher
import uvicorn

if __name__ == "__main__":
    # Run FastAPI app programmatically
    uvicorn.run("app.app_run:app", host="0.0.0.0", port=8000, reload=False)
