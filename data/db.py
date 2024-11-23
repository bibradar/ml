import psycopg as pg
import os

# Python classes
class Library:
    def __init__(self, id, name, location):
        self.id = id
        self.name = name
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
    def __init__(self, id, accesspoint_id, timestamp, user_count):
        self.id = id
        self.accesspoint_id = accesspoint_id
        self.timestamp = timestamp
        self.user_count = user_count

    def __repr__(self):
        return f"<Utilization(id={self.id}, accesspoint_id={self.accesspoint_id}, timestamp={self.timestamp}, user_count={self.user_count})>"

    
class DatabaseConnection:
    def __init__(self):
        self.connection = pg.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )

    def get_libraries(self):
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM libraries')
        libraries = cursor.fetchall()
        cursor.close()
        return [Library(*library) for library in libraries]

    def get_access_points(self):
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM access_points')
        access_points = cursor.fetchall()
        cursor.close()
        return [AccessPoint(*access_point) for access_point in access_points]

    def get_utilizations(self):
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM utilizations')
        utilizations = cursor.fetchall()
        cursor.close()
        return [Utilization(*utilization) for utilization in utilizations]
    
    def get_utilizations_by_library(self, library_id):
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM utilizations WHERE accesspoint_id IN (SELECT id FROM access_points WHERE library_id = %s)', (library_id,))
        utilizations = cursor.fetchall()
        cursor.close()
        return [Utilization(*utilization) for utilization in utilizations]

    def close(self):
        self.connection.close()