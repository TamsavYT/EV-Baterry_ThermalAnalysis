package com.example.thermalanalysis.repository;

import com.example.thermalanalysis.entity.ThermalData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ThermalDataRepository extends JpaRepository<ThermalData, Long> {
    List<ThermalData> findByBatteryId(String batteryId);
}
