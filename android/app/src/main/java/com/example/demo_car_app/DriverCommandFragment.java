package com.example.demo_car_app;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.demo_car_app.databinding.FragmentDriverCommandBinding;


public class DriverCommandFragment extends Fragment {

    private FragmentDriverCommandBinding binding;
    public static DriverCommandFragment newInstance(){
        DriverCommandFragment fragment = new DriverCommandFragment();
        return fragment;
    }

    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        binding = FragmentDriverCommandBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        MainActivity mainActivity = (MainActivity) getActivity();
        if (mainActivity != null) {
            binding.btnHorn.setOnClickListener(v -> mainActivity.sendMessage("klaxonne"));
            binding.btnHazard.setOnClickListener(v -> mainActivity.sendMessage("clignotte"));
            binding.btnUp.setOnClickListener(v -> mainActivity.sendMessage("avance"));
            binding.btnDown.setOnClickListener(v -> mainActivity.sendMessage("recule"));
            binding.btnLeft.setOnClickListener(v -> mainActivity.sendMessage("tourne_a_gauche"));
            binding.btnRight.setOnClickListener(v -> mainActivity.sendMessage("tourne_a_droite"));
        }

    }
}