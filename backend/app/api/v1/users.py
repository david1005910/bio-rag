"""User API endpoints"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbSession
from app.repositories.user import UserRepository
from app.schemas.user import UserProfile, UserResponse, UserUpdate, UsageInfo

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: CurrentUser,
) -> UserResponse:
    """
    Get current authenticated user profile

    Requires authentication.
    """
    return UserResponse.model_validate(current_user)


@router.get("/me/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: CurrentUser,
    db: DbSession,
) -> UserProfile:
    """
    Get current user profile with usage information

    Requires authentication.
    """
    user_repo = UserRepository(db)
    query_count = await user_repo.get_monthly_query_count(current_user.user_id)

    # Determine query limit based on subscription tier
    tier_limits = {
        "free": 100,
        "basic": 500,
        "premium": 2000,
        "enterprise": None,  # Unlimited
    }
    query_limit = tier_limits.get(current_user.subscription_tier)

    return UserProfile(
        user_id=current_user.user_id,
        email=current_user.email,
        name=current_user.name,
        organization=current_user.organization,
        research_fields=current_user.research_fields,
        interests=current_user.interests,
        subscription_tier=current_user.subscription_tier,
        usage=UsageInfo(
            queries_this_month=query_count,
            queries_limit=query_limit,
        ),
    )


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> UserResponse:
    """
    Update current user profile

    Only provided fields will be updated.
    Requires authentication.
    """
    user_repo = UserRepository(db)

    # Get non-None values
    update_dict = update_data.model_dump(exclude_unset=True)
    if not update_dict:
        return UserResponse.model_validate(current_user)

    updated_user = await user_repo.update(current_user.user_id, **update_dict)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse.model_validate(updated_user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """
    Delete current user account

    This action is irreversible.
    Requires authentication.
    """
    user_repo = UserRepository(db)
    success = await user_repo.delete(current_user.user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
