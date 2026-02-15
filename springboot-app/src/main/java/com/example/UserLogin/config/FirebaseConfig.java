package com.example.UserLogin.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    @PostConstruct
    public void initialize() {
        try {
            // Check if Firebase is already initialized
            if (FirebaseApp.getApps().isEmpty()) {
                // Load the service account key from resources folder
                InputStream serviceAccount = new ClassPathResource("firebase-service-account.json").getInputStream();

                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                        .setDatabaseUrl("https://ev-battery-data-default-rtdb.firebaseio.com") // Replace with your
                                                                                               // Firebase database
                        // URL
                        .build();

                FirebaseApp.initializeApp(options);
                System.out.println("Firebase initialized successfully!");
            }
        } catch (IOException e) {
            System.err.println("Error initializing Firebase: " + e.getMessage());
            e.printStackTrace();
            // You can choose to throw a RuntimeException here if Firebase is critical
            // throw new RuntimeException("Failed to initialize Firebase", e);
        }
    }
}
