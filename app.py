from src.app import app
import uvicorn
import os

if __name__ == "__main__":
    # If this is run from root, relative paths inside src/app.py 
    # need to be handled carefully. src/app.py uses os.path.dirname(__file__)
    # so it should be fine.
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
