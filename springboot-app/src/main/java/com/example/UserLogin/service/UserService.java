package com.example.UserLogin.service;

import com.example.UserLogin.entity.User;
import com.example.UserLogin.repository.UserRepository;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User register(User user) {
        return userRepository.save(user);
    }

    public String login(String email, String password) {
        User user = userRepository.findByEmail(email);
        if (user.getPassword().equals(password)) {
            return user.getName();
        } else {
            return null;
        }
    }
}
