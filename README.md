# EV Battery Thermal Analysis

This repository contains the Java Spring Boot application for EV battery thermal risk analysis.

## Project Structure

```
UserLogin/
├── Springboot-app/          # Java Spring Boot backend
│   ├── src/
│   └── pom.xml
└── data/                   # Shared datasets (Kaggle)
```

## Quick Start

### 1. Start Java Backend

```bash
cd Springboot-app
./mvnw spring-boot:run
```

Backend runs on: `http://localhost:8080`

### 2. Access Dashboard

Open browser: `http://localhost:8080/home`

## Services

- **Java Backend** - REST API, Firebase integration, serves frontend
- **Frontend Dashboard** - Real-time battery monitoring dashboard

## Documentation

- [Firebase Setup](FIREBASE_SETUP.md)

