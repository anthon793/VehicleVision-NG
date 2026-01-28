"""
Stolen Vehicle Management Routes.
Admin endpoints for managing the stolen vehicle database.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models import User
from auth import get_admin_user, get_current_user
from services.database_service import StolenVehicleService
from routes.schemas import (
    StolenVehicleCreate, StolenVehicleUpdate, StolenVehicleResponse
)
from auth.schemas import MessageResponse

router = APIRouter(prefix="/stolen-vehicles", tags=["Stolen Vehicles"])


@router.post("/", response_model=StolenVehicleResponse, status_code=status.HTTP_201_CREATED)
async def register_stolen_vehicle(
    vehicle_data: StolenVehicleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Register a new stolen vehicle (Admin only).
    
    - Plate number is normalized (uppercase, no spaces/hyphens)
    - Duplicate plate numbers are rejected
    """
    # Check for existing plate
    existing = StolenVehicleService.get_by_plate(db, vehicle_data.plate_number)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plate number already registered: {existing.plate_number}"
        )
    
    vehicle = StolenVehicleService.create(
        db=db,
        plate_number=vehicle_data.plate_number,
        vehicle_type=vehicle_data.vehicle_type.value,
        description=vehicle_data.description
    )
    
    return StolenVehicleResponse(
        id=vehicle.id,
        plate_number=vehicle.plate_number,
        vehicle_type=vehicle.vehicle_type.value,
        date_reported=vehicle.date_reported,
        description=vehicle.description,
        is_active=vehicle.is_active
    )


@router.get("/", response_model=List[StolenVehicleResponse])
async def get_all_stolen_vehicles(
    active_only: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all stolen vehicles.
    
    - By default returns only active (not resolved) records
    - Set active_only=false to include resolved vehicles
    """
    vehicles = StolenVehicleService.get_all(db, active_only=active_only)
    
    return [
        StolenVehicleResponse(
            id=v.id,
            plate_number=v.plate_number,
            vehicle_type=v.vehicle_type.value,
            date_reported=v.date_reported,
            description=v.description,
            is_active=v.is_active
        )
        for v in vehicles
    ]


@router.get("/{vehicle_id}", response_model=StolenVehicleResponse)
async def get_stolen_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific stolen vehicle by ID.
    """
    vehicle = StolenVehicleService.get_by_id(db, vehicle_id)
    
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stolen vehicle not found"
        )
    
    return StolenVehicleResponse(
        id=vehicle.id,
        plate_number=vehicle.plate_number,
        vehicle_type=vehicle.vehicle_type.value,
        date_reported=vehicle.date_reported,
        description=vehicle.description,
        is_active=vehicle.is_active
    )


@router.put("/{vehicle_id}", response_model=StolenVehicleResponse)
async def update_stolen_vehicle(
    vehicle_id: int,
    vehicle_data: StolenVehicleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Update a stolen vehicle record (Admin only).
    """
    # Build update kwargs
    update_data = vehicle_data.model_dump(exclude_unset=True)
    if "vehicle_type" in update_data and update_data["vehicle_type"]:
        update_data["vehicle_type"] = update_data["vehicle_type"].value
    
    vehicle = StolenVehicleService.update(db, vehicle_id, **update_data)
    
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stolen vehicle not found"
        )
    
    return StolenVehicleResponse(
        id=vehicle.id,
        plate_number=vehicle.plate_number,
        vehicle_type=vehicle.vehicle_type.value,
        date_reported=vehicle.date_reported,
        description=vehicle.description,
        is_active=vehicle.is_active
    )


@router.delete("/{vehicle_id}", response_model=MessageResponse)
async def delete_stolen_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Delete a stolen vehicle record (Admin only).
    """
    success = StolenVehicleService.delete(db, vehicle_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stolen vehicle not found"
        )
    
    return MessageResponse(message="Stolen vehicle record deleted successfully")


@router.post("/{vehicle_id}/resolve", response_model=StolenVehicleResponse)
async def mark_vehicle_resolved(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Mark a stolen vehicle as resolved/found (Admin only).
    """
    vehicle = StolenVehicleService.mark_resolved(db, vehicle_id)
    
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stolen vehicle not found"
        )
    
    return StolenVehicleResponse(
        id=vehicle.id,
        plate_number=vehicle.plate_number,
        vehicle_type=vehicle.vehicle_type.value,
        date_reported=vehicle.date_reported,
        description=vehicle.description,
        is_active=vehicle.is_active
    )


@router.get("/check/{plate_number}", response_model=dict)
async def check_plate_status(
    plate_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Check if a plate number is in the stolen database.
    
    Returns stolen status and vehicle details if found.
    """
    is_stolen, vehicle = StolenVehicleService.check_plate(db, plate_number)
    
    result = {
        "plate_number": plate_number.upper().replace(" ", "").replace("-", ""),
        "is_stolen": is_stolen
    }
    
    if vehicle:
        result["vehicle"] = StolenVehicleResponse(
            id=vehicle.id,
            plate_number=vehicle.plate_number,
            vehicle_type=vehicle.vehicle_type.value,
            date_reported=vehicle.date_reported,
            description=vehicle.description,
            is_active=vehicle.is_active
        ).model_dump()
    
    return result
