package com.example.demo_car_app;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.demo_car_app.databinding.FragmentAICommandBinding;
import com.example.demo_car_app.databinding.FragmentDriverCommandBinding;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link AICommandFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class AICommandFragment extends Fragment {

    private FragmentAICommandBinding binding;
    public static AICommandFragment newInstance(){
        AICommandFragment fragment = new AICommandFragment();
        return fragment;
    }

    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        binding = FragmentAICommandBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        MainActivity mainActivity = (MainActivity) getActivity();
        if (mainActivity != null) {
            binding.btnCamera.setOnClickListener(v -> mainActivity.sendMessage("camera_check"));
        }

    }
}