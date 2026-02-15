package com.example.UserLogin.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping("/dashboard")
    public String home() {
        return "battery-dashboard";
    }

    @GetMapping("/home")
    public String dashboard() {
        return "battery-dashboard";
    }
}
