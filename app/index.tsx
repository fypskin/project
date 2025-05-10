import { useState } from "react";
import { Text, View, TextInput, TouchableOpacity, Alert } from "react-native";
import { useRouter } from "expo-router";
import { supabase } from "../lib/supabase";

export default function Index() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Handle user login
  const handleLogin = async () => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      Alert.alert("Login Failed", error.message);
    } else {
      router.push("/chat"); // Redirect to chat on success
    }
  };

  // Handle user signup
  const handleSignup = async () => {
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      Alert.alert("Signup Failed", error.message);
    } else {
      Alert.alert("Success", "Check your email to confirm your account.");
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#f0f8ff", padding: 20 }}>
      <Text style={{ fontSize: 32, fontWeight: "bold", color: "#0066cc", marginBottom: 10 }}>DERMOCHAT</Text>
      <Text style={{ fontSize: 18, color: "#333", textAlign: "center", marginBottom: 30 }}>Your AI-powered skin health assistant</Text>

      {/* Email Input */}
      <TextInput
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
        style={{ width: "80%", padding: 12, borderWidth: 1, borderRadius: 10, marginBottom: 10, backgroundColor: "#fff" }}
      />

      {/* Password Input */}
      <TextInput
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        style={{ width: "80%", padding: 12, borderWidth: 1, borderRadius: 10, marginBottom: 20, backgroundColor: "#fff" }}
      />

      {/* Login Button */}
      <TouchableOpacity onPress={handleLogin} style={{ backgroundColor: "#0066cc", paddingVertical: 12, paddingHorizontal: 24, borderRadius: 25, elevation: 5, marginBottom: 10 }}>
        <Text style={{ color: "white", fontSize: 18, fontWeight: "bold" }}>Login</Text>
      </TouchableOpacity>

      {/* Signup Button */}
      <TouchableOpacity onPress={handleSignup}>
        <Text style={{ color: "#0066cc", fontSize: 16 }}>Create an Account</Text>
      </TouchableOpacity>
    </View>
  );
}
