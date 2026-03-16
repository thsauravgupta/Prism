# Project On-Device_Predictive_Model_Integration_for_SmartThings_Ecosystem

## Project Initialization

1. **Create a Samsung Account**:  
   - Visit [Samsung Account Creation](https://v3.account.samsung.com/) to create a new account.

2. **Install SmartThings App**:  
   - Download and install the SmartThings App from the [Google Play Store](https://play.google.com/store).

3. **Sign In**:  
   - Open the SmartThings App and sign in using the Samsung account you created.

4. **Refer to Demo Video**:  
   - Watch the attached video `demo_video.mp4` for instructions on creating virtual devices.

---

## Demo Video Explanation

1. **Enable Retail Mode**:  
   - Open the Life Tab and tap on it 10 times consecutively to enable retail mode. A toast message will confirm that retail mode is active.

2. **Open Devices Tab**:  
   - Navigate to the Devices Tab in the SmartThings App.

3. **Add Devices**:  
   - Click the `+` icon to add new devices.

4. **Add Multiple Devices**:  
   - Consider adding multiple devices of the following types: TV, AC, Bulb, Fridge, Washer, and Oven.

---

## Sample Android App

1. **Create a Personal Access Token**:  
   - Generate a Personal Access Token for your SmartThings account from the [SmartThings Developer Portal](https://developer.smartthings.com/docs/api/public#section/Authentication/Authorization-token-types).

2. **Build a Sample Android App**:  
   - Develop an Android app that uses the user's Personal Access Token to display a list of devices owned by the user. Refer to the [SmartThings Public APIs](https://developer.smartthings.com/docs/api/public/) for integration details.

3. **Build a Device Priority Model**:  
   - Create a model that analyzes the user's device event history and determines the order of priority for listing devices based on relevant parameters.