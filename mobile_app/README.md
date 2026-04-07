# SmartThings Devices Frontend

## Project Overview

This module is a Kotlin Android frontend built as a single-activity Jetpack Compose application. The current user flow lets an operator enter a SmartThings Personal Access Token, fetch devices from the SmartThings REST API, and inspect the returned device list. The app is intentionally small and layered so backend engineers and future automation agents can extend it without reverse-engineering implicit behavior.

At a high level, UI composables render immutable state exposed by a `ViewModel`, the `ViewModel` coordinates user actions and asynchronous work, and the repository layer owns network access. Transport details are isolated behind Retrofit interfaces so future backend integrations can be introduced without leaking HTTP concerns into UI code.

## Project Structure

- `app/src/main/java/com/example/smartthingsdevices/MainActivity.kt`: Application entry point that wires the `DeviceViewModel` into Compose.
- `app/src/main/java/com/example/smartthingsdevices/ui/DeviceListScreen.kt`: Primary Compose screen for token input, predictive-model selection, and device rendering.
- `app/src/main/java/com/example/smartthingsdevices/ui/DeviceScreenState.kt`: Immutable screen state consumed by Compose.
- `app/src/main/java/com/example/smartthingsdevices/ui/UiState.kt`: Generic async state wrapper used for loading, success, and error transitions.
- `app/src/main/java/com/example/smartthingsdevices/viewmodel/DeviceViewModel.kt`: State holder and orchestration layer for UI events.
- `app/src/main/java/com/example/smartthingsdevices/data/DeviceRepository.kt`: Repository boundary for SmartThings and future backend-facing data operations.
- `app/src/main/java/com/example/smartthingsdevices/network/SmartThingsApiService.kt`: Retrofit REST contract definitions.
- `app/src/main/java/com/example/smartthingsdevices/network/RetrofitClient.kt`: Shared OkHttp and Retrofit configuration.
- `app/src/main/java/com/example/smartthingsdevices/model/`: DTOs and model-selection contracts shared across layers.

## Tech Stack & Architecture

### UI Paradigm

- Jetpack Compose with Material 3 is the rendering model.
- `MainActivity` is the single Android entry point.
- UI elements are stateless where possible, with ephemeral view-only state kept inside composables and application state hoisted to the `ViewModel`.

### State Management

- `DeviceViewModel` is the single source of truth for screen-level state.
- `MutableStateFlow` is used internally in the `ViewModel`; `StateFlow` is exposed to the UI.
- Compose collects state through `collectAsStateWithLifecycle()` so rendering respects Android lifecycle boundaries.
- Async request status is modeled with `UiState`, while feature-level state is grouped in `DeviceScreenState`.
- Predictive model selection is hoisted into `DeviceScreenState.selectedModel` and updated only through `DeviceViewModel.selectModel(...)`.

### Concurrency

- Kotlin Coroutines power all asynchronous work.
- `viewModelScope.launch` is used for UI-triggered jobs.
- `DeviceRepository` switches network work to `Dispatchers.IO`.

### Networking

- Retrofit is the REST client.
- Gson handles JSON deserialization.
- OkHttp provides the HTTP stack and request logging.
- Authorization headers are redacted from logs in `RetrofitClient`.

## Runtime Flow

1. `MainActivity` creates `DeviceRepository` and `DeviceViewModel`.
2. `DeviceListScreen` renders `DeviceScreenState`.
3. User input triggers `DeviceViewModel.fetchDevices(...)` or `DeviceViewModel.selectModel(...)`.
4. `DeviceViewModel` calls `DeviceRepository`.
5. `DeviceRepository` invokes `SmartThingsApiService`.
6. Returned DTOs are emitted back through `StateFlow` and rendered by Compose.

## API / Backend Integration Points

### Current REST Integration

The only active backend integration today is SmartThings REST:

- Base URL: `https://api.smartthings.com/`
- Endpoint: `GET /v1/devices`
- Interface: `app/src/main/java/com/example/smartthingsdevices/network/SmartThingsApiService.kt`
- Repository entry point: `app/src/main/java/com/example/smartthingsdevices/data/DeviceRepository.kt`
- Authentication: `Authorization: Bearer <personal_access_token>`

### Incoming Payload Contract

The frontend currently expects the following JSON shape from `GET /v1/devices`:

```json
{
  "items": [
    {
      "deviceId": "device-123",
      "name": "Bedroom Lamp",
      "label": "Lamp"
    }
  ]
}
```

Mapped Kotlin models:

```kotlin
data class DeviceResponse(
    val items: List<Device>
)

data class Device(
    val deviceId: String,
    val name: String,
    val label: String
)
```

### Predictive Model Integration Surface

The frontend now tracks the user-selected predictive model in `DeviceScreenState.selectedModel`. This is the state that backend-facing features should consume when adding predictive or inference requests.

Current contract source:

```kotlin
enum class PredictiveModelOption(
    val displayName: String,
    val apiValue: String,
    val description: String
)
```

Current `apiValue` values:

- `cloud_primary`
- `on_device_optimized`
- `hybrid_balanced`

When you introduce a predictive backend endpoint, use `selectedModel.apiValue` as the request-safe identifier rather than `displayName`.

### Where New Backend or gRPC Work Should Go

- Add new transport contracts in `network/`.
- Extend or add repositories in `data/` so ViewModels stay transport-agnostic.
- Add backend DTOs in `model/`.
- Expose new UI-facing state only through `ViewModel`-owned `StateFlow`.
- If gRPC is introduced, keep the same layering: `ViewModel -> Repository -> Transport client`.

## Build & Run

From `mobile_app/`:

```bash
./gradlew assembleDebug
```

Install or run through Android Studio on an emulator or device with API 24+.

## Extension Guidelines For Backend Engineers

- Keep DTOs explicit and stable. Do not reuse UI-only types as wire contracts.
- Prefer repository methods that return domain-ready models instead of exposing Retrofit responses directly to `ViewModel`s.
- Preserve immutable screen state updates with `copy(...)`.
- Add new request parameters to repository APIs first, then thread them through the `ViewModel`, then surface them in UI.
- If authentication expands beyond a PAT, encapsulate token handling below the `ViewModel`.

## For AI Agents

- Do not mutate screen state outside of `ViewModel` classes.
- Do not perform network or disk I/O directly from composables.
- Always use coroutines for asynchronous work.
- Keep Compose UI declarative and driven by immutable state objects.
- Keep transport logic inside `data/` and `network/`, not inside `ui/`.
- Use `selectedModel.apiValue` for backend payloads and `selectedModel.displayName` only for UI text.
- Preserve bearer-token redaction in HTTP logging.
- Prefer extending existing `StateFlow`-based state models instead of introducing ad hoc mutable globals.
- Maintain the layering contract: `UI -> ViewModel -> Repository -> API client`.
