<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:orientation="vertical"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <EditText
        android:id="@+id/username_edit_text"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="Enter your username" />

    <EditText
        android:id="@+id/password_edit_text"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:inputType="textPassword"
        android:hint="Enter your password" />

    <Button
        android:id="@+id/login_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Login" />

</LinearLayout>
