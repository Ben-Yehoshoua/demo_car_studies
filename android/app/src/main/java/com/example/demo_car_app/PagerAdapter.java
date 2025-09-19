package com.example.demo_car_app;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;

public class PagerAdapter extends FragmentStateAdapter {

    public PagerAdapter(@NonNull FragmentActivity fa) {
        super(fa);
    }

    @NonNull
    @Override
    public Fragment createFragment(int position) {
        switch (position) {
            case 0:
                return new BluetoothConnectionFragment();
            case 1:
                return new DriverCommandFragment();
            case 2:
                return new AICommandFragment();
            default:
                return new DriverCommandFragment();
        }
    }

    @Override
    public int getItemCount() {
        return 3;
    }
}

