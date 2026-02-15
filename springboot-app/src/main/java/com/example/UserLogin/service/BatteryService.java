package com.example.UserLogin.service;

import com.example.UserLogin.model.BatteryData;
import com.example.UserLogin.model.BatteryData.CellData;
import com.google.firebase.database.*;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

@Service
public class BatteryService {

    private final DatabaseReference databaseReference;

    public BatteryService() {
        // Initialize Firebase Database reference
        // You can change "batteryData" to match your Firebase database structure
        this.databaseReference = FirebaseDatabase.getInstance().getReference("batteryData");
    }

    /**
     * Fetch battery data from Firebase
     * This method assumes your Firebase structure has the battery metrics at the
     * root level
     * Adjust the paths according to your actual Firebase structure
     */
    public BatteryData getBatteryData() throws ExecutionException, InterruptedException {
        System.out.println("[BatteryService] Starting to fetch battery data from Firebase");
        CompletableFuture<BatteryData> future = new CompletableFuture<>();

        databaseReference.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                try {
                    System.out.println("[BatteryService] Firebase data snapshot received");
                    System.out.println("[BatteryService] Data exists: " + dataSnapshot.exists());

                    BatteryData batteryData = new BatteryData();

                    // Parse the data from Firebase
                    if (dataSnapshot.exists()) {
                        System.out.println("[BatteryService] Parsing Firebase data...");
                        // Get main battery metrics
                        batteryData.setAverageTemperature(getDoubleValue(dataSnapshot, "averageTemperature"));
                        batteryData.setMaxTemperature(getDoubleValue(dataSnapshot, "maxTemperature"));
                        batteryData.setVoltage(getDoubleValue(dataSnapshot, "voltage"));
                        batteryData.setCurrent(getDoubleValue(dataSnapshot, "current"));
                        batteryData.setSoc(getDoubleValue(dataSnapshot, "soc"));
                        batteryData.setSoh(getDoubleValue(dataSnapshot, "soh"));
                        batteryData.setRiskLevel(getIntegerValue(dataSnapshot, "riskLevel"));
                        batteryData.setRiskStatus(getStringValue(dataSnapshot, "riskStatus"));
                        // Extract systemStatus from Firebase (active/inactive)
                        batteryData.setSystemStatus(getStringValue(dataSnapshot, "systemStatus"));
                        batteryData.setTimestamp(System.currentTimeMillis());

                        System.out.println(
                                "[BatteryService] Parsed metrics - Avg Temp: " + batteryData.getAverageTemperature());

                        // Parse cell data if available
                        DataSnapshot cellsSnapshot = dataSnapshot.child("cells");
                        if (cellsSnapshot.exists()) {
                            List<CellData> cells = new ArrayList<>();
                            for (DataSnapshot cellSnapshot : cellsSnapshot.getChildren()) {
                                CellData cell = new CellData();
                                cell.setCellId(getIntegerValue(cellSnapshot, "cellId"));
                                cell.setTemperature(getDoubleValue(cellSnapshot, "temperature"));
                                cell.setStatus(getStringValue(cellSnapshot, "status"));
                                cells.add(cell);
                            }
                            batteryData.setCells(cells);
                            System.out.println("[BatteryService] Parsed " + cells.size() + " cells from Firebase");
                        } else {
                            System.out.println("[BatteryService] No cells data found in Firebase");
                        }
                    } else {
                        // Return mock data if Firebase is empty
                        System.out.println("[BatteryService] Firebase data empty, using mock data");
                        batteryData = getMockData();
                    }

                    System.out.println("[BatteryService] Battery data ready to return");
                    future.complete(batteryData);
                } catch (Exception e) {
                    System.err.println("[BatteryService] ERROR in onDataChange: " + e.getMessage());
                    e.printStackTrace();
                    future.completeExceptionally(e);
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                System.err.println("[BatteryService] Firebase query cancelled: " + databaseError.getMessage());
                future.completeExceptionally(databaseError.toException());
            }
        });

        System.out.println("[BatteryService] Waiting for Firebase response...");
        BatteryData result = future.get();
        System.out.println("[BatteryService] Returning battery data");
        return result;
    }

    // Helper methods to safely extract values
    private Double getDoubleValue(DataSnapshot snapshot, String key) {
        Object value = snapshot.child(key).getValue();
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return null;
    }

    private Integer getIntegerValue(DataSnapshot snapshot, String key) {
        Object value = snapshot.child(key).getValue();
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return null;
    }

    private String getStringValue(DataSnapshot snapshot, String key) {
        return snapshot.child(key).getValue(String.class);
    }

    // private Boolean getBooleanValue(DataSnapshot snapshot, String key) { // Removed as per instruction
    //     Object value = snapshot.child(key).getValue();
    //     if (value instanceof Boolean) {
    //         return (Boolean) value;
    //     }
    //     return null;
    // }

    /**
     * Fallback mock data for testing when Firebase is not configured
     */
    private BatteryData getMockData() {
        System.out.println("[BatteryService] Generating mock data...");
        BatteryData batteryData = new BatteryData();
        batteryData.setAverageTemperature(35.5);
        batteryData.setMaxTemperature(42.3);
        batteryData.setVoltage(380.5);
        batteryData.setCurrent(125.0);
        batteryData.setSoc(75.0);
        batteryData.setSoh(95.0);
        batteryData.setRiskLevel(25);
        batteryData.setRiskStatus("Normal");
        batteryData.setSystemStatus("Active");
        batteryData.setTimestamp(System.currentTimeMillis());

        // Mock cell data - changed to 64 cells to match frontend
        List<CellData> cells = new ArrayList<>();
        for (int i = 1; i <= 64; i++) {
            CellData cell = new CellData();
            cell.setCellId(i);
            cell.setTemperature(30.0 + (Math.random() * 15));
            cell.setStatus("normal");
            cells.add(cell);
        }
        batteryData.setCells(cells);

        System.out.println("[BatteryService] Mock data created with " + cells.size() + " cells");
        return batteryData;
    }
}
