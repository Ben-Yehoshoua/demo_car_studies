package com.example.demo_car_app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.*;
import android.widget.*;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import java.io.IOException;
import java.io.OutputStream;
import java.util.*;

public class BluetoothConnectionFragment extends Fragment {

    private Spinner deviceSpinner;
    private ImageButton connectButton;
    private final List<BluetoothDevice> discoveredDevices = new ArrayList<>();
    private final List<String> deviceNames = new ArrayList<>();
    private ArrayAdapter<String> spinnerAdapter;

    private BluetoothAdapter bluetoothAdapter;
    private BluetoothSocket socket;
    private OutputStream outputStream;

    private static final UUID MY_UUID =
            UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    private final BroadcastReceiver bluetoothReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (BluetoothDevice.ACTION_FOUND.equals(intent.getAction())) {
                BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
                if (device != null && !discoveredDevices.contains(device)) {
                    discoveredDevices.add(device);
                    String name = (ActivityCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT)
                            != PackageManager.PERMISSION_GRANTED) ? "Appareil" : device.getName();
                    deviceNames.add((name != null ? name : "Inconnu") + " (" + device.getAddress() + ")");
                    requireActivity().runOnUiThread(() -> spinnerAdapter.notifyDataSetChanged());
                }
            }
        }
    };

    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), result -> {
                boolean allGranted = result.values().stream().allMatch(Boolean::booleanValue);
                if (allGranted) {
                    initBluetooth();
                } else {
                    Toast.makeText(requireContext(), "Permissions refusées", Toast.LENGTH_SHORT).show();
                }
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_bluetooth_connection, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        deviceSpinner = view.findViewById(R.id.deviceSpinner);
        connectButton = view.findViewById(R.id.connectButton);

        spinnerAdapter = new ArrayAdapter<>(requireContext(),
                android.R.layout.simple_spinner_item, deviceNames);
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        deviceSpinner.setAdapter(spinnerAdapter);

        connectButton.setOnClickListener(v -> connectToSelectedDevice());

        checkPermissions();
    }

    private void checkPermissions() {
        List<String> permissions = new ArrayList<>();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH_CONNECT);
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH_SCAN);
        } else {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }

        if (!permissions.isEmpty()) {
            permissionLauncher.launch(permissions.toArray(new String[0]));
        } else {
            initBluetooth();
        }
    }

    @SuppressLint("MissingPermission")
    private void initBluetooth() {
        BluetoothManager bluetoothManager = requireContext().getSystemService(BluetoothManager.class);
        bluetoothAdapter = bluetoothManager.getAdapter();

        if (bluetoothAdapter == null) {
            Toast.makeText(requireContext(), "Bluetooth non supporté", Toast.LENGTH_SHORT).show();
            return;
        }

        if (!bluetoothAdapter.isEnabled()) {
            startActivity(new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE));
        }

        requireActivity().registerReceiver(bluetoothReceiver, new IntentFilter(BluetoothDevice.ACTION_FOUND));
        bluetoothAdapter.startDiscovery();

        Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();
        for (BluetoothDevice device : pairedDevices) {
            discoveredDevices.add(device);
            deviceNames.add((device.getName() != null ? device.getName() : "Inconnu") + " (" + device.getAddress() + ")");
        }
        spinnerAdapter.notifyDataSetChanged();
    }

    @SuppressLint("MissingPermission")
    private void connectToSelectedDevice() {
        int pos = deviceSpinner.getSelectedItemPosition();
        if (pos < 0 || pos >= discoveredDevices.size()) return;

        BluetoothDevice selectedDevice = discoveredDevices.get(pos);
        Toast.makeText(requireContext(), "Connexion à " + selectedDevice.getName(), Toast.LENGTH_SHORT).show();

        new Thread(() -> {
            try {
                socket = selectedDevice.createRfcommSocketToServiceRecord(MY_UUID);
                bluetoothAdapter.cancelDiscovery();
                socket.connect();
                outputStream = socket.getOutputStream();

                // ⚠️ On transmet l'outputStream à MainActivity
                ((MainActivity) requireActivity()).setOutputStream(outputStream);

                requireActivity().runOnUiThread(() -> {
                    Toast.makeText(requireContext(), "Connecté", Toast.LENGTH_SHORT).show();
                });

            } catch (IOException e) {
                Log.e("BluetoothFragment", "Erreur de connexion", e);
                requireActivity().runOnUiThread(() -> Toast.makeText(requireContext(), "Échec de connexion", Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        try {
            requireActivity().unregisterReceiver(bluetoothReceiver);
            if (socket != null) socket.close();
        } catch (IOException | IllegalArgumentException ignored) {}
    }
}
