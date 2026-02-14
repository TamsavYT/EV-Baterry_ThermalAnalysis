package com.example.UserLogin.model;

import java.util.List;

public class BatteryData {
    private Double averageTemperature;
    private Double maxTemperature;
    private Double voltage;
    private Double current;
    private Double soc; // State of Charge
    private Double soh; // State of Health
    private Integer riskLevel;
    private String riskStatus;
    private Long timestamp;
    private List<CellData> cells;

    // Nested class for individual cell data
    public static class CellData {
        private Integer cellId;
        private Double temperature;
        private String status;

        public CellData() {
        }

        public CellData(Integer cellId, Double temperature, String status) {
            this.cellId = cellId;
            this.temperature = temperature;
            this.status = status;
        }

        // Getters and Setters
        public Integer getCellId() {
            return cellId;
        }

        public void setCellId(Integer cellId) {
            this.cellId = cellId;
        }

        public Double getTemperature() {
            return temperature;
        }

        public void setTemperature(Double temperature) {
            this.temperature = temperature;
        }

        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }
    }

    // Constructors
    public BatteryData() {
    }

    // Getters and Setters
    public Double getAverageTemperature() {
        return averageTemperature;
    }

    public void setAverageTemperature(Double averageTemperature) {
        this.averageTemperature = averageTemperature;
    }

    public Double getMaxTemperature() {
        return maxTemperature;
    }

    public void setMaxTemperature(Double maxTemperature) {
        this.maxTemperature = maxTemperature;
    }

    public Double getVoltage() {
        return voltage;
    }

    public void setVoltage(Double voltage) {
        this.voltage = voltage;
    }

    public Double getCurrent() {
        return current;
    }

    public void setCurrent(Double current) {
        this.current = current;
    }

    public Double getSoc() {
        return soc;
    }

    public void setSoc(Double soc) {
        this.soc = soc;
    }

    public Double getSoh() {
        return soh;
    }

    public void setSoh(Double soh) {
        this.soh = soh;
    }

    public Integer getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(Integer riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getRiskStatus() {
        return riskStatus;
    }

    public void setRiskStatus(String riskStatus) {
        this.riskStatus = riskStatus;
    }

    public Long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Long timestamp) {
        this.timestamp = timestamp;
    }

    public List<CellData> getCells() {
        return cells;
    }

    public void setCells(List<CellData> cells) {
        this.cells = cells;
    }
}
