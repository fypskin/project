import { Stack } from "expo-router";

export default function Layout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: "#0066cc" }, // Top bar color
        headerTintColor: "white", // Text color
        headerTitleStyle: { fontSize: 20, fontWeight: "bold" }, // Title style
        headerTitleAlign: "center", // Center align title
      }}
    />
  );
}
