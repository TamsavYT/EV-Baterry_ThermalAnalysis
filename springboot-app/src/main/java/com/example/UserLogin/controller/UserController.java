package com.example.UserLogin.controller;

import com.example.UserLogin.entity.User;
import com.example.UserLogin.service.UserService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class UserController {
    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @PostMapping("/register")
    public User register(@RequestBody User user) {
        return userService.register(user);
    }

    @PostMapping("/login")
    public String login(@RequestBody User user) {
        String name = userService.login(user.getEmail(), user.getPassword());
        if (name != null) {
            return "login successfull! welcome " + name + "!";
        } else {
            return "Login failed";
        }
    }

    @GetMapping("/error")
    public String showError() {
        return "this is error page";
    }
}
