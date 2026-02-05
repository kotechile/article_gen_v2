"""
Integration tests for user registration flow.

These tests verify the complete user registration workflow
including email verification.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestUserRegistrationFlow:
    """Test complete user registration flow."""

    def test_complete_registration_flow(self):
        """Test complete user registration flow from start to finish."""
        # Step 1: Register a new user
        registration_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["message"] == "User registered successfully. Please check your email for verification."
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["is_verified"] is False
        
        user_id = data["user"]["id"]
        
        # Step 2: Verify email (this will fail until implemented)
        verification_token = "verification_token_from_email"
        response = client.post(
            "/api/v1/auth/verify-email",
            json={"token": verification_token}
        )
        assert response.status_code == 200
        
        # Step 3: Login with verified account
        login_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123!"
        }
        
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        login_data = response.json()
        assert "access_token" in login_data
        assert "refresh_token" in login_data
        assert login_data["user"]["is_verified"] is True

    def test_registration_with_existing_email(self):
        """Test registration with existing email address."""
        # First registration
        registration_data = {
            "email": "existing@example.com",
            "password": "SecurePass123!",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 201
        
        # Second registration with same email
        response = client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 409

    def test_registration_validation(self):
        """Test registration with invalid data."""
        # Test weak password
        weak_password_data = {
            "email": "test@example.com",
            "password": "weak",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/auth/register", json=weak_password_data)
        assert response.status_code == 422
        
        # Test invalid email
        invalid_email_data = {
            "email": "invalid-email",
            "password": "SecurePass123!",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/auth/register", json=invalid_email_data)
        assert response.status_code == 422
        
        # Test missing required fields
        incomplete_data = {
            "email": "test@example.com",
            "password": "SecurePass123!"
            # Missing first_name and last_name
        }
        
        response = client.post("/api/v1/auth/register", json=incomplete_data)
        assert response.status_code == 422

    def test_registration_with_special_characters(self):
        """Test registration with special characters in names."""
        registration_data = {
            "email": "special@example.com",
            "password": "SecurePass123!",
            "first_name": "José",
            "last_name": "O'Connor-Smith"
        }
        
        response = client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["user"]["first_name"] == "José"
        assert data["user"]["last_name"] == "O'Connor-Smith"

    def test_registration_email_case_insensitive(self):
        """Test that email addresses are case insensitive."""
        registration_data = {
            "email": "Test@Example.COM",
            "password": "SecurePass123!",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 201
        
        # Try to register with different case
        registration_data_lower = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "first_name": "Jane",
            "last_name": "Smith"
        }
        
        response = client.post("/api/v1/auth/register", json=registration_data_lower)
        assert response.status_code == 409  # Should conflict
