import psycopg as pg
import os


class Library:
    def __init__(self, id, name, bib, uni, location):
        self.id = id
        self.name = name
        self.bib = bib
        self.uni = uni
        self.location = location

    def __repr__(self):
        return f"<Library(id={self.id}, name={self.name}, location={self.location})>"


class AccessPoint:
    def __init__(self, id, name, library_id):
        self.id = id
        self.name = name
        self.library_id = library_id

    def __repr__(self):
        return f"<AccessPoint(id={self.id}, name={self.name}, library_id={self.library_id})>"


class Utilization:
    def __init__(self, accesspoint_id, timestamp, user_count=None):
        self.accesspoint_id = accesspoint_id
        self.timestamp = timestamp
        self.user_count = user_count

    def __repr__(self):
        return f"<Utilization(id={self.id}, accesspoint_id={self.accesspoint_id}, timestamp={self.timestamp}, user_count={self.user_count})>"


class AggregateUtilization:
    def __init__(self, timestamp, user_count):
        self.timestamp = timestamp
        self.user_count = user_count

    def __repr__(self):
        return f"<AggregateUtilization(timestamp={self.timestamp}, user_count={self.user_count})>"


class DatabaseConnection:
    def __init__(self):
        self.connection = pg.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
        print("Connected to database")

    def get_libraries(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM library")
        libraries = cursor.fetchall()
        cursor.close()
        return [Library(*library) for library in libraries]


    def get_access_points(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM accesspoint")
        access_points = cursor.fetchall()
        cursor.close()
        return [AccessPoint(*access_point) for access_point in access_points]

    def get_utilizations(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM utilization")
        utilizations = cursor.fetchall()
        cursor.close()
        return [Utilization(*utilization) for utilization in utilizations]

    def get_utilizations_by_library(self, library_id):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                        SELECT u.timestamp, SUM(u.user_count)
                        FROM utilization u
                        JOIN accesspoint ON u.accesspoint_id = accesspoint.id
                        WHERE accesspoint.library_id = %s
                        GROUP BY u.timestamp
                        ORDER BY u.timestamp
                       """,
            (library_id,),
        )
        utilizations = cursor.fetchall()
        cursor.close()
        return [AggregateUtilization(*utilization) for utilization in utilizations]
    
    def get_max_count_for_library(self, library_id: int):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                SELECT cast(max(percentile_cont) as integer) from user_count_stats
                WHERE id = %s
                group by id
            """,
            (library_id,),
        )
        utilization = cursor.fetchone()
        cursor.close()
        return utilization[0]
        

    def get_user_count_stats_of_day(self, day: int):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                        SELECT * from user_count_stats
                        where day = %s
                       """,
            (day,),
        )
        stats = cursor.fetchall()
        cursor.close()
        return stats

    def get_one_day(self, library_id: int, timestamp: int):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                SELECT 
                    SUM(U.user_count)  
                FROM 
                    Utilization U
                JOIN 
                    AccessPoint A ON U.accesspoint_id = A.id
                WHERE 
                    A.library_id = %s
                    AND U.timestamp >= to_timestamp(%s)
                    AND U.timestamp < to_timestamp(%s) + INTERVAL '24 HOURS'
                GROUP BY U.timestamp
                ORDER BY U.timestamp;
            """,
            (library_id, timestamp, timestamp),
        )
        stats = cursor.fetchall()
        cursor.close()
        return stats
        
    
    def get_user_count_with_timestamp(self, library_id: int, timestamp: int):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                        SELECT sum(u.user_count)
                        FROM utilization u
                        JOIN accesspoint ON u.accesspoint_id = accesspoint.id
                        WHERE accesspoint.library_id = %s
                        AND u.timestamp = to_timestamp(%s)
                       """,
            (library_id, timestamp),
        )
        stats = cursor.fetchone()
        cursor.close()
        return stats

    def close(self):
        self.connection.close()
