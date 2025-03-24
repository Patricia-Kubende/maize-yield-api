from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ Create SQLite database
DATABASE_URL = "sqlite:///./maize_yield.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# ✅ Base class for ORM models
Base = declarative_base()

# ✅ User Table for Authentication
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # 🔒 Hashed Password will be stored

# ✅ Predictions Table to Store Past Predictions
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)  # 🆔 Link predictions to users
    Soil_Type = Column(String, nullable=False)
    pH = Column(Float, nullable=False)
    Seed_Variety = Column(String, nullable=False)
    Rainfall_mm = Column(Float, nullable=False)
    Temperature_C = Column(Float, nullable=False)
    Humidity_percent = Column(Float, nullable=False)
    Planting_Date = Column(String, nullable=False)
    Fertilizer_Type = Column(String, nullable=False)
    Predicted_Yield = Column(Float, nullable=False)
    Confidence_Range = Column(String, nullable=False)
    Category = Column(String, nullable=False)
    Recommendation = Column(String, nullable=False)

# ✅ Create tables in the database
Base.metadata.create_all(bind=engine)

# ✅ Database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
