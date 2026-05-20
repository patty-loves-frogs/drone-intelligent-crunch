"""
Mock image generator for testing the drone dashboard.
Creates synthetic test images with various scenarios.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random


def create_frames_dir():
    """Ensure frames directory exists."""
    if not os.path.exists("frames"):
        os.makedirs("frames")
        print("✅ Created frames/ directory")


def create_person_on_ground(filename: str):
    """Create a test image with a person on the ground."""
    img = Image.new('RGB', (800, 600), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Ground (green)
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    
    # Person (simple stick figure)
    # Head
    draw.ellipse([350, 200, 450, 300], fill=(255, 200, 124))
    # Body
    draw.rectangle([390, 300, 410, 400], fill=(50, 50, 50))
    # Arms
    draw.rectangle([350, 320, 450, 340], fill=(255, 200, 124))
    # Legs
    draw.rectangle([385, 400, 395, 480], fill=(50, 50, 50))
    draw.rectangle([405, 400, 415, 480], fill=(50, 50, 50))
    
    # Label
    draw.text((10, 10), "PERSON ON GROUND", fill=(255, 0, 0))
    
    img.save(f"frames/{filename}")
    print(f"✅ Created {filename}")


def create_vehicle(filename: str):
    """Create a test image with a vehicle."""
    img = Image.new('RGB', (800, 600), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(100, 100, 100))  # Gray asphalt
    
    # Vehicle (car)
    draw.rectangle([250, 350, 550, 420], fill=(255, 0, 0))  # Car body
    draw.rectangle([270, 330, 350, 350], fill=(100, 100, 100))  # Front window
    draw.rectangle([450, 330, 530, 350], fill=(100, 100, 100))  # Back window
    # Wheels
    draw.ellipse([270, 415, 330, 475], fill=(0, 0, 0))
    draw.ellipse([470, 415, 530, 475], fill=(0, 0, 0))
    
    # Label
    draw.text((10, 10), "VEHICLE DETECTED", fill=(255, 255, 0))
    
    img.save(f"frames/{filename}")
    print(f"✅ Created {filename}")


def create_group_of_people(filename: str):
    """Create a test image with multiple people."""
    img = Image.new('RGB', (800, 600), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    
    # Group of people (simplified)
    positions = [(250, 200), (350, 220), (450, 200), (550, 230)]
    
    for x, y in positions:
        # Head
        draw.ellipse([x, y, x+40, y+40], fill=(255, 200, 124))
        # Body
        draw.rectangle([x+15, y+40, x+25, y+80], fill=(50, 50, 50))
    
    # Label
    draw.text((10, 10), "GROUP OF PEOPLE", fill=(255, 0, 0))
    
    img.save(f"frames/{filename}")
    print(f"✅ Created {filename}")


def create_empty_scene(filename: str):
    """Create a test image with no objects."""
    img = Image.new('RGB', (800, 600), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    
    # Add some clouds (white circles)
    draw.ellipse([100, 50, 150, 100], fill=(255, 255, 255))
    draw.ellipse([500, 100, 550, 150], fill=(255, 255, 255))
    
    # Label
    draw.text((10, 10), "EMPTY SCENE", fill=(0, 255, 0))
    
    img.save(f"frames/{filename}")
    print(f"✅ Created {filename}")


def create_mixed_scenario(filename: str):
    """Create a test image with multiple elements."""
    img = Image.new('RGB', (800, 600), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(100, 100, 100))
    
    # Vehicle
    draw.rectangle([100, 320, 300, 380], fill=(0, 0, 255))
    draw.ellipse([110, 375, 150, 415], fill=(0, 0, 0))
    draw.ellipse([250, 375, 290, 415], fill=(0, 0, 0))
    
    # Person near vehicle
    draw.ellipse([350, 250, 400, 300], fill=(255, 200, 124))
    draw.rectangle([370, 300, 380, 350], fill=(50, 50, 50))
    
    # Label
    draw.text((10, 10), "VEHICLE + PERSON", fill=(255, 255, 0))
    
    img.save(f"frames/{filename}")
    print(f"✅ Created {filename}")


def generate_all_mocks():
    """Generate all mock images."""
    print("🎨 Generating mock test images...\n")
    
    create_frames_dir()
    
    print("Creating test scenarios:\n")
    create_person_on_ground("01_person_ground.png")
    create_vehicle("02_vehicle.png")
    create_group_of_people("03_group_people.png")
    create_empty_scene("04_empty_scene.png")
    create_mixed_scenario("05_vehicle_person.png")
    
    # Create some duplicates with variations for testing YOLO
    create_person_on_ground("06_person_standing.png")
    create_vehicle("07_car_parked.png")
    
    print("\n✅ All mock images generated!")
    print(f"📁 Images saved in: frames/")
    print(f"🧪 Ready to test the dashboard!")


if __name__ == "__main__":
    generate_all_mocks()
