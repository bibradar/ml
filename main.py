from dotenv import load_dotenv, dotenv_values

from data.db import DatabaseConnection
from data.get_data import get_data_frame, load_model_and_get_prediction
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import uvicorn
import datetime
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path=".env")

db = DatabaseConnection()
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LibraryScorePredictionInput(BaseModel):
    library_id: int
    arrival_time: int

class LibraryScorePredictionOutput(BaseModel):
    library_id: int
    score: float

class LibraryOccupancyPredictionOutput(BaseModel):
    library_id: int
    occupancy: str



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/libraries")
def get_libraries():
    return db.get_libraries()

@app.get("/libraries_day_prediction", response_model=List[LibraryOccupancyPredictionOutput])
def get_libraries_day_prediction():
    libraries = db.get_libraries()
    prediction = load_model_and_get_prediction("2024-11-23 12:00:00")
    print(prediction)
    predictions = [
        LibraryOccupancyPredictionOutput(
            library_id=library.id,
            occupancy=str(prediction)
        ) for library in libraries
    ]
    return predictions


@app.get("/user_count_stats/{day}")
def get_user_count_stats_of_day(day: int):
    stats_by_lib = {}
    for (
        lib_id,
        day,
        hour,
        avg_user_count,
        max_user_count,
    ) in db.get_user_count_stats_of_day(day):
        if lib_id not in stats_by_lib:
            stats_by_lib[lib_id] = []

        stats_by_lib[lib_id].append(
            {
                "avg_user_count": avg_user_count,
                "max_user_count": max_user_count,
            }
        )

    return stats_by_lib


@app.post("/predict", response_model=List[LibraryScorePredictionOutput])
def predict(input_data: List[LibraryScorePredictionInput]):
    print(input_data)

    predictions = []
    max_time = 0
    for library in input_data:
        time_to_library = library.arrival_time - int(datetime.datetime.now().timestamp())
        if time_to_library > max_time:
            max_time = time_to_library

    for library in input_data:
        # 0. Time to get to library (arrival_time - now)
        time_to_library = library.arrival_time - int(time.time())

        # 1. Get the predicted user count for the library at the given arrival time
        data = get_data_frame(library.library_id)
        # prediction_user_precentage = model.predict(data, time_to_library)
        prediction_user_percentage = load_model_and_get_prediction(data, library.arrival_time)

        prediction_user_percentage = 0.5  # value how many users  are predicted to be in the library at the given time

        # 2. Weight the predicted user count and the distance to the library to a score
        normalized_time = time_to_library / max_time

        weight_time = 0.5
        weight_user_percentage = 0.5

        # 3. Return the libraries sorted by the score
        score = (weight_time * (1 - normalized_time)) + (
            weight_user_percentage * (1 - prediction_user_percentage)
        )
        predictions = []

        prediction = LibraryScorePredictionOutput(
            library_id=library.library_id,
            score=score,
            stats={
                "time_to_library": time_to_library,
                "predicted_user_percentage": prediction_user_percentage,
            },
        )
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
