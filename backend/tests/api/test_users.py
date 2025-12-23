"""Tests for user API endpoints"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestGetCurrentUser:
    """Tests for getting current user"""

    async def test_get_me_authenticated(
        self, client: AsyncClient, auth_headers: dict, test_user
    ):
        """Test getting current user when authenticated"""
        response = await client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name

    async def test_get_me_unauthenticated(self, client: AsyncClient):
        """Test getting current user without authentication"""
        response = await client.get("/api/v1/users/me")
        assert response.status_code == 401  # No auth header


class TestUpdateUser:
    """Tests for updating user"""

    async def test_update_name(
        self, client: AsyncClient, auth_headers: dict, test_user
    ):
        """Test updating user name"""
        response = await client.patch(
            "/api/v1/users/me",
            headers=auth_headers,
            json={"name": "Updated Name"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    async def test_update_organization(
        self, client: AsyncClient, auth_headers: dict, test_user
    ):
        """Test updating organization"""
        response = await client.patch(
            "/api/v1/users/me",
            headers=auth_headers,
            json={"organization": "New Organization"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["organization"] == "New Organization"

    async def test_update_multiple_fields(
        self, client: AsyncClient, auth_headers: dict, test_user
    ):
        """Test updating multiple fields"""
        response = await client.patch(
            "/api/v1/users/me",
            headers=auth_headers,
            json={
                "name": "New Name",
                "organization": "New Org",
                "research_fields": ["oncology", "immunology"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        assert data["organization"] == "New Org"
        assert "oncology" in data["research_fields"]


class TestDeleteUser:
    """Tests for deleting user"""

    async def test_delete_user(
        self, client: AsyncClient, auth_headers: dict, test_user
    ):
        """Test deleting current user"""
        response = await client.delete("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 204

        # Verify user is deleted
        response = await client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 401  # Token no longer valid
