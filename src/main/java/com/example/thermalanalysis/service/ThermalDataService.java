package com.example.thermalanalysis.service;

import com.example.thermalanalysis.entity.ThermalData;
import com.example.thermalanalysis.repository.ThermalDataRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ThermalDataService {
    private final ThermalDataRepository thermalDataRepository;

    public ThermalDataService(ThermalDataRepository thermalDataRepository) {
        this.thermalDataRepository = thermalDataRepository;
    }

    public ThermalData saveData(ThermalData data) {
        return thermalDataRepository.save(data);
    }

    public List<ThermalData> getByBatteryId(String batteryId) {
        return thermalDataRepository.findByBatteryId(batteryId);
    }
}
