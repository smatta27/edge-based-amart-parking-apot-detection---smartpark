import cv2
import json
import argparse

spots = []
drawing = False

ix, iy = -1, -1


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, spots, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            # Draw all previous rectangles
            for spot in spots:
                cv2.rectangle(img_copy, (spot[0], spot[1]), (spot[2], spot[3]), (0, 255, 0), 2)
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        spots.append([x1, y1, x2, y2])
        print(f"Spot added: {[x1, y1, x2, y2]}")
        img_copy = img.copy()
        for spot in spots:
            cv2.rectangle(img_copy, (spot[0], spot[1]), (spot[2], spot[3]), (0, 255, 0), 2)


def redraw_all_rectangles():
    global img_copy, spots
    img_copy = img.copy()
    for spot in spots:
        cv2.rectangle(img_copy, (spot[0], spot[1]), (spot[2], spot[3]), (0, 255, 0), 2)


def main():
    global img, img_copy, spots
    parser = argparse.ArgumentParser(description='Select parking spots manually')
    parser.add_argument('image_path', type=str, help='Path to the parking lot image')
    parser.add_argument('--output', type=str, default='spots.json', help='Output JSON file for spot coordinates')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not read image from {args.image_path}")
        return
    img_copy = img.copy()

    print("Instructions:")
    print("Draw rectangles for each parking spot by clicking and dragging.")
    print("Press 's' to save and exit, 'r' to reset, 'u' to undo last, or 'q' to quit without saving.")

    cv2.namedWindow('Select Parking Spots')
    cv2.setMouseCallback('Select Parking Spots', draw_rectangle)

    while True:
        cv2.imshow('Select Parking Spots', img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save spots to JSON
            with open(args.output, 'w') as f:
                json.dump(spots, f)
            print(f"Saved {len(spots)} spots to {args.output}")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break
        elif key == ord('r'):
            spots = []
            img_copy = img.copy()
            print("All spots cleared. Start over.")
        elif key == ord('u'):
            if spots:
                removed = spots.pop()
                redraw_all_rectangles()
                print(f"Removed last spot: {removed}")
            else:
                print("No spots to undo.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
