from dotenv import load_dotenv, dotenv_values

from data.db import DatabaseConnection
from data.get_data import get_data_frame, get_model, get_max_user_count, predict_one_day, load_model_and_get_prediction, get_count_from_last_week, load_model_and_get_prediction2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import uvicorn
import datetime
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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
    stats: Dict[str, Any]

class LibraryOccupancyPredictionOutput(BaseModel):
    library_id: int
    occupancy: list[int]



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/max_count")
def read_max(library_id: int):
    return get_max_user_count(library_id)    


@app.get("/libraries")
def get_libraries():
    return db.get_libraries()
    
def average_every_hour(occ):
    return [int(sum(map(lambda e: e['predicted_user_count'], occ[i:i+4])) / 4) for i in range(0, len(occ), 4)]

@app.get("/libraries_day_prediction", response_model=List[LibraryOccupancyPredictionOutput])
def get_libraries_day_prediction():
    
    libraries = db.get_libraries()
    current_day_timestamp = int(datetime.datetime.now().replace(hour=0, minute=0, second=0).timestamp())

    # preds = map(lambda l: (l.id, load_model_and_get_prediction(current_day_timestamp, l.id)), libraries)
    preds = map(lambda l: (l.id, load_model_and_get_prediction2(current_day_timestamp, l.id)), libraries)

    output =  [
        LibraryOccupancyPredictionOutput(
            library_id=library_id,
            occupancy=average_every_hour(occ)
        ) for library_id, occ in preds
    ]
    
    return output


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
            stats_by_lib[lib_id] = {"avg_user_count": [], "max_user_count": []}

        # stats_by_lib[lib_id].append(
        #     {
        #         "avg_user_count": avg_user_count,
        #         "max_user_count": max_user_count,
        #     }
        # )

        stats_by_lib[lib_id]["avg_user_count"].append(avg_user_count)
        stats_by_lib[lib_id]["max_user_count"].append(max_user_count)



    return stats_by_lib

@app.post("/predictt", response_model=List[LibraryScorePredictionOutput])
def predict(input_data: List[LibraryScorePredictionInput]):
    
    predictions = []
    max_time = 0
    for library in input_data:
        time_to_library = library.arrival_time - int(datetime.datetime.now().timestamp())

        if time_to_library < 0:
            raise HTTPException(status_code=400, detail="arrival_time is in the past")
        if time_to_library > max_time:
            max_time = time_to_library


    for library in input_data:
        time_to_library = library.arrival_time - int(datetime.datetime.now().timestamp())

        # if library.library_id == 1:
        #     predictions = load_model_and_get_prediction(library.arrival_time, library.library_id)

        count = get_count_from_last_week(library.arrival_time, library.library_id)
        max_count = get_max_user_count(library.library_id)
        
        # 2. Weight the predicted user count and the distance to the library to a score
        normalized_time = time_to_library / max_time
        relative_count = count / max_count

        weight_time = 0.7
        weight_user_percentage = 0.3

        # 3. Return the libraries sorted by the score
        score = (weight_time * (1 - normalized_time)) + (
            weight_user_percentage * (1 - relative_count)
        )

        prediction = LibraryScorePredictionOutput(
            library_id=library.library_id,
            score=score,
            stats={
                "time_to_library": time_to_library,
                "predicted_user_percentage": count,
            },
        )
        
        predictions.append(prediction)
        
    return predictions
    


@app.post("/predict", response_model=List[LibraryScorePredictionOutput])
def predict(input_data: List[LibraryScorePredictionInput]):
    print(input_data)

    predictions = []
    max_time = 0
    for library in input_data:
        time_to_library = library.arrival_time - int(datetime.datetime.now().timestamp())

        if time_to_library < 0:
            raise HTTPException(status_code=400, detail="arrival_time is in the past")
        if time_to_library > max_time:
            max_time = time_to_library

    for library in input_data:
        # 0. Time to get to library (arrival_time - now)
        time_to_library = library.arrival_time - int(datetime.datetime.now().timestamp())
        
        models_predictions = load_model_and_get_prediction(library.arrival_time, library.library_id)

        timestamp = pd.Timestamp(library.arrival_time, unit='s', tz='Europe/Berlin')
        pred = next(filter(lambda p: p['timestamp'] == timestamp, models_predictions), None)
        if pred == None:
            raise HTTPException(status_code=400, detail="No prediction available for this timestamp")
    
        # 2. Weight the predicted user count and the distance to the library to a score
        normalized_time = time_to_library / max_time

        max_user_count = get_max_user_count(library.library_id)
        relative_count = pred['predicted_user_count'] / max_user_count

        weight_time = 0.7
        weight_user_percentage = 0.3

        # 3. Return the libraries sorted by the score
        score = (weight_time * (1 - normalized_time)) + (
            weight_user_percentage * (1 - relative_count)
        )

        prediction = LibraryScorePredictionOutput(
            library_id=library.library_id,
            score=score,
            stats={
                "time_to_library": time_to_library,
                "predicted_user_percentage": pred['predicted_user_count'],
            },
        )
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
