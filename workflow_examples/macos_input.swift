import Cocoa
import CoreGraphics
import Foundation

enum InputAction: String {
    case move = "move"
    case click = "click"
    case moveClick = "move-click"
    case screenGeometry = "screen-geometry"
}

func usage() -> Never {
    fputs("Usage: swift macos_input.swift <move|click|move-click> <x> <y>\n", stderr)
    fputs("   or: swift macos_input.swift screen-geometry\n", stderr)
    exit(2)
}

func parseCoordinate(_ value: String) -> CGFloat {
    guard let parsed = Double(value) else {
        fputs("Invalid coordinate: \(value)\n", stderr)
        exit(2)
    }
    return CGFloat(parsed)
}

func makePoint(x: CGFloat, y: CGFloat) -> CGPoint {
    let screenHeight = NSScreen.main?.frame.height ?? 0
    return CGPoint(x: x, y: screenHeight - y)
}

func emitScreenGeometry() -> Never {
    guard let screen = NSScreen.main else {
        fputs("Failed to resolve main screen\n", stderr)
        exit(1)
    }

    let frame = screen.frame
    let scale = screen.backingScaleFactor
    let payload: [String: Any] = [
        "width_points": Int(round(frame.width)),
        "height_points": Int(round(frame.height)),
        "width_pixels": Int(round(frame.width * scale)),
        "height_pixels": Int(round(frame.height * scale)),
        "scale_factor": scale,
    ]

    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        guard let output = String(data: data, encoding: .utf8) else {
            fputs("Failed to encode screen geometry output\n", stderr)
            exit(1)
        }
        print(output)
        exit(0)
    } catch {
        fputs("Failed to serialize screen geometry: \(error)\n", stderr)
        exit(1)
    }
}

func setCursorPosition(_ point: CGPoint) {
    CGWarpMouseCursorPosition(point)
    CGAssociateMouseAndMouseCursorPosition(1)
}

func postMouseEvent(
    type: CGEventType,
    point: CGPoint,
    button: CGMouseButton = .left,
    clickState: Int64 = 1
) {
    guard let event = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: point, mouseButton: button) else {
        fputs("Failed to create mouse event\n", stderr)
        exit(1)
    }
    event.setIntegerValueField(.mouseEventClickState, value: clickState)
    event.post(tap: .cghidEventTap)
}

let arguments = CommandLine.arguments
guard arguments.count == 2 || arguments.count == 4 else { usage() }

guard let action = InputAction(rawValue: arguments[1]) else {
    usage()
}

if action == .screenGeometry {
    guard arguments.count == 2 else { usage() }
    emitScreenGeometry()
}

let x = parseCoordinate(arguments[2])
let y = parseCoordinate(arguments[3])
let point = makePoint(x: x, y: y)

switch action {
case .move:
    setCursorPosition(point)
    usleep(50_000)
    postMouseEvent(type: .mouseMoved, point: point)
case .click:
    setCursorPosition(point)
    usleep(80_000)
    postMouseEvent(type: .mouseMoved, point: point)
    usleep(40_000)
    postMouseEvent(type: .leftMouseDown, point: point)
    usleep(30_000)
    postMouseEvent(type: .leftMouseUp, point: point)
case .moveClick:
    setCursorPosition(point)
    usleep(80_000)
    postMouseEvent(type: .mouseMoved, point: point)
    usleep(40_000)
    postMouseEvent(type: .leftMouseDown, point: point)
    usleep(30_000)
    postMouseEvent(type: .leftMouseUp, point: point)
case .screenGeometry:
    emitScreenGeometry()
}
