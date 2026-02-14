package com.example.UserLogin.controller;

import com.example.UserLogin.model.BatteryData;
import com.example.UserLogin.service.BatteryService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // Enable CORS for frontend access
public class BatteryController {

    private final BatteryService batteryService;

    public BatteryController(BatteryService batteryService) {
        this.batteryService = batteryService;
    }

    /**
     * GET endpoint to fetch battery data
     * Accessible at: http://localhost:8080/api/battery-data
     */
    @GetMapping("/battery-data")
    public ResponseEntity<BatteryData> getBatteryData() {
        System.out.println("========================================");
        System.out.println("[BatteryController] Received request for battery data");
        try {
            BatteryData batteryData = batteryService.getBatteryData();
            System.out.println("[BatteryController] Successfully fetched battery data");
            System.out.println("[BatteryController] Risk Level: " + batteryData.getRiskLevel());
            System.out.println("[BatteryController] Avg Temp: " + batteryData.getAverageTemperature());
            System.out.println("[BatteryController] Number of cells: "
                    + (batteryData.getCells() != null ? batteryData.getCells().size() : 0));
            System.out.println("[BatteryController] Returning data to client");
            System.out.println("========================================");
            return ResponseEntity.ok(batteryData);
        } catch (Exception e) {
            System.err.println("[BatteryController] ERROR fetching battery data: " + e.getMessage());
            e.printStackTrace();
            System.out.println("========================================");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
