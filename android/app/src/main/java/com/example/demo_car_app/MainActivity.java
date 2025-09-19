package com.example.demo_car_app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothSocket;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.Toast;

import com.example.demo_car_app.R;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import androidx.viewpager2.widget.ViewPager2;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;


public class MainActivity extends AppCompatActivity {

    private OutputStream outputStream;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ViewPager2 viewPager = findViewById(R.id.viewPager);
        PagerAdapter adapter = new PagerAdapter(this);
        viewPager.setAdapter(adapter);
    }

    public void setOutputStream(OutputStream os) {
        this.outputStream = os;

    }

    public void sendMessage(String message) {
        if (outputStream == null || message.isEmpty()) return;

        new Thread(() -> {
            try {
                outputStream.write(message.getBytes());
                runOnUiThread(() ->
                        Toast.makeText(this, "Message envoyé", Toast.LENGTH_SHORT).show()
                );
            } catch (IOException e) {
                runOnUiThread(() ->
                        Toast.makeText(this, "Erreur d'envoi", Toast.LENGTH_SHORT).show()
                );
                e.printStackTrace();
            }
        }).start();
    }

}
