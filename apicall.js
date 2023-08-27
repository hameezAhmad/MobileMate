

// import React, { useEffect, useState } from 'react';
// import { StyleSheet, Text, View } from 'react-native';

// export default function App() {
//   const [message, setMessage] = useState('');

//   useEffect(() => {
//     fetch('http://192.168.8.205:5000/')
//       .then(response => response.json())
//       .then(data => setMessage(data.message))
//       .catch(error => console.error(error));
//   }, []);

//   return (
//     <View style={styles.container}>
//       <Text>{message}</Text>
//     </View>
//   );
// }

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#fff',
//     alignItems: 'center',
//     justifyContent: 'center',
//   },
// });n