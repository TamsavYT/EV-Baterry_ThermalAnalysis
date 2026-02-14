# Firebase Setup Instructions

## Prerequisites
You need to set up Firebase to use this application. Follow these steps:

## Step 1: Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or select an existing project
3. Follow the setup wizard

## Step 2: Enable Realtime Database
1. In your Firebase project, go to "Build" → "Realtime Database"
2. Click "Create Database"
3. Choose a location
4. Start in "Test mode" (you can secure it later)

## Step 3: Get Service Account Key
1. Go to Project Settings (gear icon) → "Service accounts"
2. Click "Generate new private key"
3. Download the JSON file
4. **Rename it to:** `firebase-service-account.json`
5. **Place it in:** `src/main/resources/firebase-service-account.json`

## Step 4: Update Firebase Configuration
Open `src/main/java/com/example/UserLogin/config/FirebaseConfig.java` and update:

```java
.setDatabaseUrl("https://YOUR-PROJECT-ID.firebaseio.com")
```

Replace `YOUR-PROJECT-ID` with your actual Firebase project ID (found in Project Settings).

## Step 5: Firebase Data Structure
Your Firebase Realtime Database should have this structure:

```json
{
  "batteryData": {
    "averageTemperature": 35.5,
    "maxTemperature": 42.3,
    "voltage": 380.5,
    "current": 125.0,
    "soc": 75,
    "soh": 95,
    "riskLevel": 25,
    "riskStatus": "Normal",
    "cells": {
      "0": {
        "cellId": 1,
        "temperature": 35.2,
        "status": "normal"
      },
      "1": {
        "cellId": 2,
        "temperature": 36.8,
        "status": "normal"
      }
      // ... more cells
    }
  }
}
```

**Note:** If Firebase is not configured or empty, the application will use mock data automatically.

## Step 6: Test the Setup
1. Start your Spring Boot application
2. Open browser and go to: `http://localhost:8080/api/battery-data`
3. You should see JSON data (either from Firebase or mock data)
4. Go to: `http://localhost:8080/home` to see the dashboard

## Security (Important for Production)
Before deploying to production:
1. Update Firebase Database Rules
2. Never commit `firebase-service-account.json` to version control
3. Add `firebase-service-account.json` to `.gitignore`
