package com.example.thermalanalysis.controller;

import com.example.thermalanalysis.entity.ThermalData;
import com.example.thermalanalysis.service.ThermalDataService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ThermalAnalysisController {
    private final ThermalDataService thermalDataService;

    public ThermalAnalysisController(ThermalDataService thermalDataService) {
        this.thermalDataService = thermalDataService;
    }

    @PostMapping("/submit-data")
    public ThermalData submitData(@RequestBody ThermalData data) {
        return thermalDataService.saveData(data);
    }

    @GetMapping("/")
    public String status() {
        return "Thermal Analysis app is running";
    }

    @GetMapping("/error")
    public String showError() {
        return "this is error page";
    }
}
