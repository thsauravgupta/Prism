package com.example.smartthingsdevices.model

/**
 * Maps SmartSense device-type indices (0–37) to human-readable names.
 * Sourced from Two_level_Arch/src/config.py DEFAULT_DEVICE_TYPES.
 */
object DeviceTypes {

    private val INDEX_TO_NAME = mapOf(
        0 to "AirConditioner",
        1 to "AirPurifier",
        2 to "Blind",
        3 to "Camera",
        4 to "ClothingCareMachine",
        5 to "Computer",
        6 to "ContactSensor",
        7 to "Dishwasher",
        8 to "DoorBell",
        9 to "Dryer",
        10 to "Elevator",
        11 to "Fan",
        12 to "GarageDoor",
        13 to "GasValve",
        14 to "Humidifier",
        15 to "LeakSensor",
        16 to "Light",
        17 to "Microwave",
        18 to "MotionSensor",
        19 to "MultiFunctionalSensor",
        20 to "NetworkAudio",
        21 to "None",
        22 to "Other",
        23 to "PresenceSensor",
        24 to "Projector",
        25 to "Refrigerator",
        26 to "RemoteController",
        27 to "RobotCleaner",
        28 to "SetTop",
        29 to "Siren",
        30 to "SmartLock",
        31 to "SmartPlug",
        32 to "Switch",
        33 to "Television",
        34 to "Thermostat",
        35 to "Vent",
        36 to "Washer",
        37 to "WaterValve"
    )

    /** Total number of device types in the SmartSense dataset. */
    const val NUM_DEVICE_TYPES = 38

    /** Returns the human-readable name for the given device index. */
    fun nameFor(index: Int): String =
        INDEX_TO_NAME[index] ?: "Device $index"
}
