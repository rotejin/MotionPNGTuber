
import numpy as np

# Standard 68-point mapping (0-indexed)
# Mouth: 48-67
#   Outer lip: 48-59
#   Inner lip: 60-67
#
# Anime-face-detector (28 points) often used:
#   0-3: Right Eye
#   4-7: Left Eye
#   8-10: Nose
#   11-22: Face components? 
#   24-27: Mouth (Top, Bot, Left, Right)? 
#
# Let's verify standard 68 points for Mouth.
# 48: Left Corner
# 54: Right Corner
# 51: Top Center (Outer)
# 57: Bottom Center (Outer)
#
# If the user's code uses [24, 25, 26, 27] as MOUTH_OUTLINE, and assumes they are 4 points.
# I need to see if those 4 points correspond to [Left, Top, Right, Bottom] or similar.

def main():
    print("Investigation of Landmark Mapping")
    print("-" * 30)
    
    # Standard 68 points indices for "Diamond" mouth (4 points)
    # Left, Top, Right, Bottom
    mouth_diamond_68 = [48, 51, 54, 57]
    print(f"Standard 68-point mouth diamond (Outer): {mouth_diamond_68}")
    
    # Eyes for rotation
    # Standard 68:
    # Right Eye: 36-41 (digits are standard, R/L depends on viewer vs subject. Standard is Subject's Right = 36-41)
    # Left Eye: 42-47
    
    # Wait, usually 36-41 is Right Eye (on Viewer's Left) and 42-47 is Left Eye (on Viewer's Right)?
    # Actually dlib/standard 68:
    # 0-16: Jaw
    # 17-21: Right Brow
    # 22-26: Left Brow
    # 27-35: Nose
    # 36-41: Right Eye
    # 42-47: Left Eye
    # 48-67: Mouth
    
    print("Eyes (Standard 68):")
    print("  Right Eye (Subject's): 36-41")
    print("  Left Eye (Subject's): 42-47")
    
    # Determine what 24-27 was in original code.
    # Original code: "MOUTH_OUTLINE = [24, 25, 26, 27]"
    # It calculates `xy.mean(axis=0)` -> center.
    # It calculates width/height from min/max of these 4 points.
    
    print("\nProposed Mapping for Replacement:")
    print("  Mouth (4 pts) -> [48, 54, 51, 57] (Left, Right, Top, Bot)")
    print("  Or use full outer lip [48-59] to get better bounds.")
    
    print("\nStatus: CONFIRMED feasibility if 'dwpose' outputs standard 68 face points.")

if __name__ == "__main__":
    main()
